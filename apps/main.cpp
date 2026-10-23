#include <VideoReader.h>
#include <Vision.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc.hpp>


namespace
{
//根据测量得到的灯条中心、长宽和倾角绘制检测框
cv::Mat draw_light_bars(const cv::Mat& image, const std::vector<LightBar>& light_bars)
{
    //只在原图副本上叠框，原图和用于检测的二值图保持不变
    cv::Mat display_image{image.clone()};

    for (const LightBar& light_bar : light_bars)
    {
        const double angle_radians{light_bar.angle * CV_PI / 180.0};

        const cv::Point2f half_length{
            static_cast<float>(std::sin(angle_radians) * light_bar.length / 2.0),
            static_cast<float>(-std::cos(angle_radians) * light_bar.length / 2.0)
        };

        const cv::Point2f half_width{
            static_cast<float>(std::cos(angle_radians) * light_bar.width / 2.0),
            static_cast<float>(std::sin(angle_radians) * light_bar.width / 2.0)
        };

        const std::array<cv::Point2f, 4> corners{
            light_bar.center + half_length + half_width,
            light_bar.center + half_length - half_width,
            light_bar.center - half_length - half_width,
            light_bar.center - half_length + half_width
        };

        for (std::size_t index{0}; index < corners.size(); ++index)
        {
            cv::line(display_image, corners[index], corners[(index + 1) % corners.size()],
                     cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

    }

    return display_image;
}


void draw_armors(cv::Mat& display_image, const std::vector<Armor>& armors)
{
    for (const Armor& armor : armors)
    {
        for (std::size_t index{0}; index < armor.corners.size(); ++index)
        {
            cv::line(display_image, armor.corners[index], armor.corners[(index + 1) % armor.corners.size()],
                     cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
        }
    }
}


//回调只记录用户请求，实际定位由读取线程完成
struct SeekRequest
{
    int target{-1};
    int displayed_frame{};
    bool updating_trackbar{false};
};


void on_frame_changed(int position, void* userdata)
{
    SeekRequest& request{*static_cast<SeekRequest*>(userdata)};
    if (!request.updating_trackbar && position != request.displayed_frame)
    {
        request.target = position;
    }
}

}


int main()
{
    try
    {
        const std::string video_path{"drone_blue3.avi"};

        //创建对象的同时构造并启动取流线程
        VideoReader video_reader{video_path};
        const VideoInfo& video_info{video_reader.get_info()};
        const int last_frame{video_info.frame_count - 1};

        std::cout << "视频：" << video_path
                  << "\n分辨率：" << video_info.width << " × " << video_info.height
                  << "，共 " << video_info.frame_count << " 帧，" << video_info.fps << " FPS\n"
                  << "空格：暂停/继续；拖动Frame滑条：跳转并暂停；关闭窗口：退出。\n"
                  << "按源FPS播放，处理落后时跳帧；绿色为灯条，橙色为当前帧装甲板。\n";


        //预处理、单灯条筛选和配对参数统一在Vision.h中调整。
        const PreprocessParams preprocess_params{};  //预处理
        const LightBarFilterParams light_bar_params{};  //单灯条筛选
        const ArmorPairParams pair_params{};  //配对参数

        const ImagePreprocessor image_preprocessor{preprocess_params};
        const ArmorDetector armor_detector{light_bar_params, pair_params};

        //判断是否能成功取帧
        VideoFrame processing_frame{};
        if (!video_reader.get_frame(processing_frame))
        {
            throw std::runtime_error("无法读取视频第一帧");
        }

        //创建窗口和滑条
        const std::string window_name{"Task5"};
        const std::string trackbar_name{"Frame"};
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, 960, 768);

        SeekRequest seek{};
        if (last_frame > 0)
        {
            cv::createTrackbar(
                trackbar_name, 
                window_name, 
                nullptr, 
                last_frame, 
                on_frame_changed, 
                &seek
            );
        }


        int frame_index{processing_frame.source_index};  //当前帧编号
        int requested_frame{-1};  //滑条移动的目标帧号
        bool awaiting_frame{false};  //是否追赶到目标帧
        bool paused{last_frame == 0};  //是否暂停
        bool redraw{true};  //是否需要处理帧
        
        auto playback_start{std::chrono::steady_clock::now()};  //用于计算播放进度
        int playback_start_frame{frame_index};


        while (true)
        {
            if (redraw)
            {
                const cv::Mat binary_image{
                    image_preprocessor.preprocess(processing_frame.image)
                };  //预处理 

                const std::vector<LightBar> light_bars{
                    armor_detector.find_light_bars(binary_image)
                };  //筛选并测量

                const std::vector<Armor> armors{
                    armor_detector.pair_light_bars(light_bars)
                };  //择优匹配

                cv::Mat display_image{
                    draw_light_bars(processing_frame.image, light_bars)
                };  //在原图副本上绘制灯条框

                draw_armors(display_image, armors);  //在原图副本上绘制装甲板

                cv::imshow(window_name, display_image);

                //同步当前帧号和滑条
                seek.displayed_frame = frame_index;
                if (last_frame > 0)
                {
                    seek.updating_trackbar = true;
                    cv::setTrackbarPos(trackbar_name, window_name, frame_index);
                    seek.updating_trackbar = false;
                }

                //在获取到下一帧前不再处理
                redraw = false;
            }

            //暂停时仍处理窗口事件；播放时只等待下一帧时刻前的剩余时间。
            int wait_ms{20};

            if (!paused)
            {
                const double passed_seconds{
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - playback_start
                    ).count()
                };  //计算已耗费的时间

                const double next_frame_time{(frame_index - playback_start_frame + 1) / video_info.fps};
                
                wait_ms = std::clamp(
                    static_cast<int>((next_frame_time - passed_seconds) * 1000.0), 
                    1, 
                    20
                );  //选择等待的最佳时间
            }

            //等待时的选项
            const int key{cv::waitKey(wait_ms)};

            if (cvGetWindowHandle(window_name.c_str()) == nullptr)  //判断窗口是否存在
            {
                break;
            }

            if (seek.target >= 0)  //接收滑块跳转请求并暂停
            {
                requested_frame = seek.target;
                seek.target = -1;

                video_reader.seek(requested_frame);

                awaiting_frame = false;
                paused = true;
            }

            if (key == ' ')  //空格暂停操作
            {
                paused = !paused;

                if (!paused && requested_frame < 0)
                {
                    paused = frame_index == last_frame;
                    if (!paused)
                    {
                        //丢弃暂停前尚未显示的预取帧，从当前帧的下一帧继续。
                        video_reader.seek(frame_index + 1);
                        awaiting_frame = false;
                    }
                }

                playback_start = std::chrono::steady_clock::now();  //重新建立播放起点
                playback_start_frame = requested_frame >= 0 ? requested_frame : frame_index;
            }

            int target{frame_index};
            
            if (!paused)
            {
                const double passed_seconds{
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - playback_start
                    ).count()
                };  //计算从playback_start到现在用时多久

                const int elapsed_frames{static_cast<int>(std::min(passed_seconds * video_info.fps,
                    static_cast<double>(video_info.frame_count - playback_start_frame)))};

                target = std::min(playback_start_frame + elapsed_frames, last_frame);

                if (requested_frame < 0 && target > frame_index && !awaiting_frame)
                {
                    video_reader.skip_to(target);
                    //等待本次解码完成再追赶，避免持续推高目标而丢掉所有正在解码的帧。
                    awaiting_frame = true;
                }
            }

            //解码尚未完成时继续轮询窗口，不能阻塞关闭、暂停或下一次滑条请求。
            VideoFrame next_frame{};

            if ((requested_frame >= 0 || (!paused && target > frame_index))
                && video_reader.try_get_frame(next_frame))
            {
                processing_frame = std::move(next_frame);
                frame_index = processing_frame.source_index;
                requested_frame = -1;
                awaiting_frame = false;
                redraw = true;

                if (frame_index == last_frame)
                {
                    paused = true;
                }
            }
        }

        video_reader.stop();
        cv::destroyAllWindows();
        std::cout << "读取停止\n";
    }
    catch (const std::exception& ex)
    {
        cv::destroyAllWindows();
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}
