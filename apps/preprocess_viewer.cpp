#include <Vision.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


//调参查看器：取帧 → 共用预处理 → 单灯条筛选 → 加权评分择优配对 → 对照显示。
//原图、纯二值图和检测结果分窗显示；默认每根灯条只参与一组配对，不做跨帧保留。
//参数统一在Vision.h中的PreprocessParams、LightBarFilterParams和ArmorPairParams调整；预处理见ImagePreprocessor。
//修改源码后需要重新编译并重启，本程序没有实时HSV调参滑条，Frame滑条只用于选择帧。
//从项目根目录执行：
//cmake --build build --target preprocess_viewer --parallel 2
//./build/preprocess_viewer
//导出完整带框视频（不打开窗口）：
//./build/preprocess_viewer --export-video [输出路径.mp4]


namespace
{

//回调和主循环之间传递跳转请求；回调只记录帧号，实际读取交给主循环处理。
struct SeekRequest
{
    int target{-1};  //待跳转的源帧号；-1表示没有请求，0表示视频第一帧
    int displayed_frame{};  //当前窗口显示的源帧号，用于取消拖回当前位置的请求
    bool updating_trackbar{false};  //区分程序更新滑条与用户拖动，避免播放时触发跳转
};


void on_frame_changed(int position, void* userdata)
{
    //createTrackbar把&seek作为userdata传入；这里取回同一个对象以记录用户操作
    SeekRequest& request{*static_cast<SeekRequest*>(userdata)};

    //程序推进滑条只更新显示；用户改变位置才提出跳转请求
    if (!request.updating_trackbar)
    {
        request.target = position != request.displayed_frame ? position : -1;
    }
}


//按从0开始的源帧号读取。成功才替换image，失败保留旧图并通过error说明原因。
//位置检查依赖视频后端报告的帧号，不能单靠这些检查证明任意视频的跳转像素完全准确。
bool read_frame(cv::VideoCapture& capture, int target, cv::Mat& image, std::string& error)
{
    //顺序播放时不反复seek；只有回退或跳转才重新定位
    if (static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES)) != target)
    {
        if (!capture.set(cv::CAP_PROP_POS_FRAMES, target))
        {
            error = "视频后端不支持跳转到帧 " + std::to_string(target);
            return false;
        }
    }

    if (static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES)) != target)
    {
        error = "视频后端未定位到请求帧 " + std::to_string(target);
        return false;
    }

    cv::Mat decoded_image{};

    if (!capture.read(decoded_image) || decoded_image.empty())
    {
        error = "无法解码帧 " + std::to_string(target);
        return false;
    }

    //POS_FRAMES表示下一次要读取的位置，成功读取后应为当前帧号加一
    if (static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES)) != target + 1)
    {
        error = "读取后的帧位置与请求不符，未替换当前图像";
        return false;
    }

    image = decoded_image;
    return true;
}


//生成专供imshow使用的显示副本；保存时仍使用未加底栏的image和mask。
cv::Mat make_display(const cv::Mat& image, int frame_index, int frame_count,
                     double fps, bool paused, const std::string& message)
{
    //信息放在额外边栏，不遮挡原图内容；原始图像和二值数据均不修改
    cv::Mat display{};
    cv::copyMakeBorder(image, display, 0, 92, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(24));

    std::ostringstream status{};
    status << "Frame " << frame_index << '/' << frame_count - 1
           << " | Time " << std::fixed << std::setprecision(3) << frame_index / fps
           << " s | " << (paused ? "PAUSED" : "PLAYING");

    if (frame_index == frame_count - 1)
    {
        status << " | END";
    }

    //HighGUI内置字体不支持中文，窗口用英文短提示，终端和导出说明用中文
    cv::putText(display, status.str(), {12, image.rows + 24},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar::all(255), 1, cv::LINE_AA);
    cv::putText(display, "Space: play/pause | A/D: previous/next | S: save | Q/Esc: quit",
                {12, image.rows + 51}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar::all(255), 1, cv::LINE_AA);
    cv::putText(display, message, {12, image.rows + 78}, cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar::all(255), 1, cv::LINE_AA);

    return display;
}


//只绘制副本，不把框写入预处理数据；绿色为单灯条候选，橙色为当前模式选出的配对。
cv::Mat draw_detection(const cv::Mat& image, const std::vector<LightBar>& light_bars,
                       const std::vector<Armor>& armors)
{
    cv::Mat display{image.clone()};

    for (const LightBar& light_bar : light_bars)
    {
        //LightBar保存相对竖直方向的倾角；以短边为宽、长边为高还原对应的旋转矩形。
        const cv::RotatedRect rectangle{
            light_bar.center, cv::Size2f{light_bar.width, light_bar.length}, light_bar.angle
        };
        cv::Point2f corners[4]{};
        rectangle.points(corners);

        for (int index{0}; index < 4; ++index)
        {
            cv::line(display, corners[index], corners[(index + 1) % 4], cv::Scalar{0, 255, 0}, 1);
        }
    }

    for (const Armor& armor : armors)
    {
        for (std::size_t index{0}; index < armor.corners.size(); ++index)
        {
            cv::line(display, armor.corners[index], armor.corners[(index + 1) % armor.corners.size()],
                     cv::Scalar{0, 165, 255}, 2);
        }
    }

    return display;
}


//把同一帧的原图、单通道二值图和HSV记录保存为一组，便于固定帧号比较不同参数。
//mask保留原始分辨率和0/255像素值；文件名中的时间标记用于区分导出批次，不是视频时间。
std::filesystem::path save_frame(const std::string& video_path, const cv::Mat& image,
                               const cv::Mat& mask, int frame_index, double fps,
                               const PreprocessParams& params, const std::filesystem::path& directory)
{
    std::filesystem::create_directories(directory);

    const auto timestamp{std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()};
    std::ostringstream filename{};
    filename << "frame_" << std::setfill('0') << std::setw(6) << frame_index << '_' << timestamp;

    std::filesystem::path prefix{directory / filename.str()};
    int duplicate{};

    //同一帧可反复导出，已有同名文件时增加编号，不覆盖此前材料
    while (std::filesystem::exists(prefix.string() + "_original.png")
           || std::filesystem::exists(prefix.string() + "_mask.png")
           || std::filesystem::exists(prefix.string() + "_info.txt"))
    {
        prefix = directory / (filename.str() + '_' + std::to_string(++duplicate));
    }

    const std::string original_path{prefix.string() + "_original.png"};
    const std::string mask_path{prefix.string() + "_mask.png"};
    const std::string info_path{prefix.string() + "_info.txt"};

    if (!cv::imwrite(original_path, image) || !cv::imwrite(mask_path, mask))
    {
        throw std::runtime_error("图像保存失败，部分文件可能已写入：" + prefix.string());
    }

    std::ofstream info{info_path};
    info << "视频：" << std::filesystem::absolute(video_path).string()
         << "\n源帧号（从0开始）：" << frame_index
         << "\n视频FPS：" << fps
         << "\n视频时间（源帧号/FPS，秒）：" << std::fixed << std::setprecision(6) << frame_index / fps
         << "\n分辨率：" << image.cols << " × " << image.rows
         << "\n第一段HSV下限：" << params.low_range.lower_bound
         << "\n第一段HSV上限：" << params.low_range.upper_bound
         << "\n第二段HSV下限：" << params.high_range.lower_bound
         << "\n第二段HSV上限：" << params.high_range.upper_bound
         << "\n高斯核：" << params.gaussian_kernel << "，sigma：" << params.gaussian_sigma
         << "\n闭运算矩形核：" << params.closing_kernel << "，轮数：1"
         << "\n预处理：共用ImagePreprocessor，高斯滤波→HSV双区间合并→闭运算"
         << "\n原图：" << std::filesystem::absolute(original_path).string()
         << "\n二值图：" << std::filesystem::absolute(mask_path).string()
         << "\n二值图为原始分辨率单通道数据，不包含窗口文字或检测标记。\n";
    info.close();

    if (!info)
    {
        throw std::runtime_error("图像已写入，但参数记录保存失败：" + info_path);
    }

    std::cout << "已保存当前帧 " << frame_index << "：\n"
              << std::filesystem::absolute(original_path).string() << '\n'
              << std::filesystem::absolute(mask_path).string() << '\n'
              << std::filesystem::absolute(info_path).string() << '\n';

    return prefix;
}

//按源帧顺序独立导出，复用查看器的参数和绘制，不把暂停、跳转或窗口底栏写入视频。
void export_detection_video(cv::VideoCapture& capture, double fps, int frame_count,
                            const PreprocessParams& params, const ArmorDetector& detector,
                            const std::filesystem::path& output_path)
{
    if (output_path.extension() != ".mp4")
    {
        throw std::runtime_error("导出路径需要以.mp4结尾");
    }

    //未完成时保留.partial.mp4，全部写完并关闭编码器后才改为正式文件名。
    const std::filesystem::path partial_path{
        output_path.parent_path() / (output_path.stem().string() + ".partial.mp4")
    };

    if (std::filesystem::exists(output_path) || std::filesystem::exists(partial_path))
    {
        throw std::runtime_error("输出文件或未完成文件已存在，请更换导出路径：" + output_path.string());
    }

    if (!output_path.parent_path().empty())
    {
        std::filesystem::create_directories(output_path.parent_path());
    }

    const ImagePreprocessor image_preprocessor{params};
    cv::VideoWriter writer{};
    int written_frames{};

    try
    {
        cv::Mat image{};
        std::string error{};

        if (!read_frame(capture, 0, image, error))
        {
            throw std::runtime_error(error);
        }

        //使用源视频尺寸和FPS；编码耗时不会改变输出视频的播放时长。
        const cv::Size frame_size{image.size()};
        writer.open(partial_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps, frame_size, true);

        if (!writer.isOpened())
        {
            throw std::runtime_error("无法创建MP4视频，请检查输出位置及OpenCV的mp4v编码支持");
        }

        std::cout << "开始导出带框视频：" << std::filesystem::absolute(output_path).string()
                  << "\n分辨率：" << frame_size.width << " × " << frame_size.height
                  << "，FPS：" << fps << "，总帧数：" << frame_count << '\n' << std::flush;

        for (int frame_index{}; frame_index < frame_count; ++frame_index)
        {
            if (frame_index > 0 && !read_frame(capture, frame_index, image, error))
            {
                throw std::runtime_error(error);
            }

            if (image.size() != frame_size || image.type() != CV_8UC3)
            {
                throw std::runtime_error("源帧尺寸或类型不符合视频写入要求，帧号："
                                         + std::to_string(frame_index));
            }

            const cv::Mat mask{image_preprocessor.preprocess(image)};
            const std::vector<LightBar> light_bars{detector.find_light_bars(mask)};
            const std::vector<Armor> armors{detector.pair_light_bars(light_bars)};
            writer.write(draw_detection(image, light_bars, armors));
            ++written_frames;

            if (written_frames % 500 == 0 || written_frames == frame_count)
            {
                std::cout << "已处理 " << written_frames << '/' << frame_count << " 帧\n" << std::flush;
            }
        }

        //release写入容器尾部信息；在此之前文件还不能视为完整视频。
        writer.release();

        if (std::filesystem::file_size(partial_path) == 0)
        {
            throw std::runtime_error("编码器未生成有效的视频数据");
        }

        std::filesystem::rename(partial_path, output_path);
    }
    catch (const std::exception& error)
    {
        writer.release();
        throw std::runtime_error("导出失败，已处理 " + std::to_string(written_frames)
            + " 帧：" + error.what() + "\n若已生成，未完成文件位于："
            + std::filesystem::absolute(partial_path).string());
    }

    std::cout << "导出完成：" << std::filesystem::absolute(output_path).string()
              << "\n共 " << written_frames << " 帧，时长 " << written_frames / fps << " 秒。\n";
}

}


int main(int argc, char* argv[])
{
    try
    {
        const bool export_video{argc >= 2 && std::string{argv[1]} == "--export-video"};

        if ((argc == 2 && std::string{argv[1]} == "--help")
            || (argc > 1 && (!export_video || argc > 3)))
        {
            std::cout << "查看器：./build/preprocess_viewer\n"
                      << "导出带框视频：./build/preprocess_viewer --export-video [输出路径.mp4]\n"
                      << "未指定路径时，自动保存到preprocess_exports/；已有文件不会覆盖。\n";
            return argc == 2 && std::string{argv[1]} == "--help" ? 0 : 1;
        }

        //素材入口：相对路径按启动时的工作目录解析，所以应从项目根目录启动。
        //更换素材可改这里的路径；与主程序的video_path是各自独立的设置。
        const std::string video_path{"drone_blue3.avi"};
        //查看器自行读取视频以支持回退和任意跳转，不使用主程序的多线程VideoReader。
        cv::VideoCapture capture{video_path};

        if (!capture.isOpened())
        {
            throw std::runtime_error("无法打开视频，请从项目根目录启动：" + video_path);
        }

        const double fps{capture.get(cv::CAP_PROP_FPS)};
        const int frame_count{static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT))};

        if (fps <= 0.0 || frame_count <= 0)
        {
            throw std::runtime_error("视频帧率或总帧数无效，无法建立播放进度条");
        }

        //==================== 共用参数入口 ====================
        //HSV双区间、高斯核和闭运算核在Vision.h的PreprocessParams中集中定义。
        //两个Scalar分别为下限和上限，内部顺序都是H、S、V：
        //H（色相）：当前8位HSV取0～179，对应通常色相角度的一半；控制保留哪段颜色。
        //S（饱和度）：取0～255；降低下限可接纳颜色较淡的区域，也可能引入灰白背景。
        //V（明度）：取0～255；降低下限可接纳较暗的区域，也可能引入暗处干扰。
        //三个分量都在各自上下限内才保留为255（白），否则为0（黑）；上下限包含边界。
        //普通区间中，每项下限应不大于上限；H下限大于上限不会自动跨越0连接两个区间。
        //先固定同一帧，每次只改一个分量的一侧边界，检查真实灯条是否完整留下，再看干扰。
        //这里和主程序读取同一组默认参数；修改Vision.h后重新编译并启动才会生效。
        const PreprocessParams preprocess_params{};
        const LightBarFilterParams light_bar_params{};
        const ArmorPairParams pair_params{};
        const ImagePreprocessor image_preprocessor{preprocess_params};
        //首帧、播放、逐帧、定位和视频导出共用同一个检测器。
        const ArmorDetector detector{light_bar_params, pair_params};

        std::cout << "灯条测量：白色区域主轴（无法测量则跳过）"
                  << "\n面积筛选：白色像素数量"
                  << "，范围：" << light_bar_params.min_area << "～" << light_bar_params.max_area
                  << "\n配对模式：加权评分择优，每根灯条最多使用一次"
                  << "\n评分权重（方向差、相对长度差、纵向错位）："
                  << pair_params.angle_weight << "、" << pair_params.length_weight << "、"
                  << pair_params.vertical_offset_weight << '\n';

        if (export_video)
        {
            const auto timestamp{std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()};
            const std::filesystem::path output_path{argc == 3 ? std::filesystem::path{argv[2]}
                : std::filesystem::path{"preprocess_exports"} / ("detection_" + std::to_string(timestamp) + ".mp4")};
            export_detection_video(capture, fps, frame_count, preprocess_params, detector, output_path);
            return 0;
        }

        cv::Mat image{};
        std::string error{};

        if (!read_frame(capture, 0, image, error))
        {
            throw std::runtime_error(error);
        }

        //检测使用纯二值图，几何合格组合按加权评分择优并执行灯条独占。
        //首帧和后续帧使用同样的三步处理；结果只来自当前帧，不保留历史框。
        cv::Mat mask{image_preprocessor.preprocess(image)};
        std::vector<LightBar> light_bars{detector.find_light_bars(mask)};
        std::vector<Armor> armors{detector.pair_light_bars(light_bars)};
        int frame_index{};  //当前image和mask对应的源帧号，从0开始
        bool paused{frame_count == 1};  //多帧视频默认播放，单帧视频直接停留
        bool redraw{true};  //图像或状态变化时才重新生成显示副本
        std::string message{"Drag Frame to seek and pause."};
        SeekRequest seek{};
        const std::string original_window{"Preprocess - 初始视频"};
        const std::string mask_window{"Preprocess - 二值化"};
        const std::string detection_window{"Preprocess - 筛选并配对"};
        const std::string trackbar_name{"Frame"};

        //960×768只控制窗口初始显示尺寸，预处理和导出仍使用原始分辨率。
        cv::namedWindow(original_window, cv::WINDOW_NORMAL);
        cv::resizeWindow(original_window, 960, 768);
        cv::namedWindow(mask_window, cv::WINDOW_NORMAL);
        cv::resizeWindow(mask_window, 960, 768);
        cv::namedWindow(detection_window, cv::WINDOW_NORMAL);
        cv::resizeWindow(detection_window, 960, 768);

        if (frame_count > 1)
        {
            //范围0～frame_count-1；只建立帧选择滑条，不控制任何预处理参数。
            cv::createTrackbar(trackbar_name, original_window, nullptr, frame_count - 1, on_frame_changed, &seek);
        }

        std::cout << "预处理查看器：" << video_path << "，共 " << frame_count << " 帧，" << fps << " FPS\n"
                  << "空格：暂停/继续；a/d：上一帧/下一帧；拖动原图窗口的Frame进度条：跳转并暂停。\n"
                  << "s：保存当前原图、二值图和HSV记录到preprocess_exports/；q/Esc或关闭窗口：退出。\n"
                  << "检测结果窗口：绿色为单灯条，橙色为当前配对模式的结果；不做历史保留。\n"
                  << "自动播放按源FPS计时，处理落后时跳帧；暂停后仍可逐帧检查。\n"
                  << "帧号从0开始；播放到结尾停留，仍可回看和保存。\n";

        //以恢复播放时的帧号和时间为起点，避免把每帧处理耗时额外加到视频时长上。
        auto playback_start{std::chrono::steady_clock::now()};
        int playback_start_frame{frame_index};

        while (true)
        {
            if (redraw)
            {
                cv::imshow(original_window, make_display(image, frame_index, frame_count, fps, paused, message));
                cv::imshow(mask_window, make_display(mask, frame_index, frame_count, fps, paused, message));
                const std::string detection_message{
                    "Green: " + std::to_string(light_bars.size()) + " light bars | Orange: "
                    + std::to_string(armors.size()) + " pairs (exclusive bars)"
                };
                cv::imshow(detection_window, make_display(draw_detection(image, light_bars, armors),
                    frame_index, frame_count, fps, paused, detection_message));
                seek.displayed_frame = frame_index;

                if (frame_count > 1)
                {
                    seek.updating_trackbar = true;
                    cv::setTrackbarPos(trackbar_name, original_window, frame_index);
                    seek.updating_trackbar = false;
                }

                redraw = false;
            }

            //waitKey同时处理窗口事件和按键；暂停时仍需轮询，才能响应滑条、保存及关闭。
            //按键前需让图像窗口获得焦点；20是暂停时的轮询等待毫秒数，不是视频帧间隔。
            int wait_ms{20};

            if (!paused)
            {
                const double elapsed{std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - playback_start).count()};
                const double next_frame_time{(frame_index - playback_start_frame + 1) / fps};
                wait_ms = std::clamp(static_cast<int>((next_frame_time - elapsed) * 1000.0), 1, 20);
            }

            const int key{cv::waitKey(wait_ms)};

            //沿用工程GTK后端的窗口检测方式，不依赖其未实现的WND_PROP_VISIBLE
            if (key == 27 || key == 'q' || key == 'Q'
                || cvGetWindowHandle(original_window.c_str()) == nullptr
                || cvGetWindowHandle(mask_window.c_str()) == nullptr
                || cvGetWindowHandle(detection_window.c_str()) == nullptr)
            {
                break;
            }

            if (key == 's' || key == 'S')
            {
                try
                {
                    //保存当前实际显示帧对应的原数据；目录相对于启动工作目录，按s时才创建。
                    save_frame(video_path, image, mask, frame_index, fps, preprocess_params, "preprocess_exports");
                    message = "Saved frame " + std::to_string(frame_index) + " (paths in terminal).";
                }
                catch (const std::exception& save_error)
                {
                    message = "Save failed (see terminal).";
                    std::cerr << "保存失败：" << save_error.what() << '\n';
                }

                redraw = true;
            }

            int target{-1};  //本轮准备读取的帧号；与seek.target一样，-1表示不读取新帧

            //先接收滑条请求；同一轮若又按a/d，则以下方逐帧按键确定的目标为准。
            if (seek.target >= 0)
            {
                target = seek.target;
                seek.target = -1;
                paused = true;
                redraw = true;
            }

            if (key == ' ')
            {
                paused = !paused;
                redraw = true;
                playback_start = std::chrono::steady_clock::now();
                playback_start_frame = target >= 0 ? target : frame_index;
            }

            if (key == 'a' || key == 'A' || key == 'd' || key == 'D')
            {
                const int step{key == 'a' || key == 'A' ? -1 : 1};
                //clamp把目标限制在首末帧之间，避免首帧回退或末帧前进导致越界。
                target = std::clamp(frame_index + step, 0, frame_count - 1);
                paused = true;
                redraw = true;
            }

            //自动播放按实际时间选帧；处理落后就跳过中间帧，手动定位和a/d不受影响。
            bool automatic_frame{false};

            if (target < 0 && !paused)
            {
                const double elapsed{std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - playback_start).count()};
                const int elapsed_frames{static_cast<int>(std::min(elapsed * fps,
                    static_cast<double>(frame_count - playback_start_frame)))};
                const int scheduled_frame{playback_start_frame + elapsed_frames};

                if (scheduled_frame > frame_index)
                {
                    //即使已经落后到视频结束，也先显示末帧，再沿用原来的末帧暂停处理。
                    target = std::min(scheduled_frame, frame_count - 1);
                    automatic_frame = true;

                    if (target == frame_index)
                    {
                        target = frame_count;
                    }
                }
            }

            if (target >= frame_count)
            {
                //到结尾停留而不退出，方便回退、比较和保存末尾目标。
                paused = true;
                redraw = true;
                message = "End of video. Seek or press A to review.";
            }
            else if (target >= 0 && target != frame_index)
            {
                //跳过的帧只grab、不取出图像和检测；避免每次追赶都随机seek视频。
                if (automatic_frame)
                {
                    int next_frame{static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES))};

                    while (next_frame < target)
                    {
                        if (!capture.grab())
                        {
                            throw std::runtime_error("自动播放跳帧时读取失败，源帧号：" + std::to_string(next_frame));
                        }

                        ++next_frame;
                    }
                }

                if (read_frame(capture, target, image, error))
                {
                    frame_index = target;
                    //定位、逐帧和自动播放都重新计算当前帧，避免跳转后残留前一帧检测结果。
                    mask = image_preprocessor.preprocess(image);
                    light_bars = detector.find_light_bars(mask);
                    armors = detector.pair_light_bars(light_bars);
                    message = "Drag Frame to seek and pause.";

                    if (frame_index == frame_count - 1)
                    {
                        paused = true;
                        message = "End of video. Seek or press A to review.";
                    }
                }
                else
                {
                    //读取失败时不更新帧号和mask，继续显示原来那一帧，避免图像与标号错配。
                    paused = true;
                    message = "Read failed; keeping previous frame (see terminal).";
                    std::cerr << error << '\n';
                }

                redraw = true;
            }
        }

        cv::destroyAllWindows();
    }
    catch (const std::exception& error)
    {
        cv::destroyAllWindows();
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
