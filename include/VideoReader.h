#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

//视频的属性
struct VideoInfo
{
    int width{};
    int height{};
    double fps{};
    int frame_count{};
};

//帧的属性，相较于cv::Mat更详细
struct VideoFrame
{
    cv::Mat image{};
    int source_index{};  //原视频帧号
};


class VideoReader
{
public:
    explicit VideoReader(
        const std::string& video_path,
        int frame_step = 1,
        std::size_t image_node_num = 3);  //构造时打开视频并启动读取线程

    ~VideoReader();

    VideoReader(const VideoReader&) = delete;  //禁止拷贝
    VideoReader& operator=(const VideoReader&) = delete;

    const VideoInfo& get_info() const;

    //等待下一帧；自然读完后先取尽队列，再返回 false
    bool get_frame(VideoFrame& output_frame);

    //不等待解码，暂无新帧就返回false，让主线程继续响应窗口事件。
    bool try_get_frame(VideoFrame& output_frame);

    //滑条定位：清空旧缓存，由读取线程定位；正在解码的旧帧也不能再入队。
    void seek(int source_index);

    //自动播放追赶时间：丢弃目标之前的帧，读取线程只grab跳过它们。
    void skip_to(int source_index);

    //提前取消读取并等待线程退出；停止后不再交付队列中的剩余帧
    void stop();

private:
    void read_loop();

    //构造函数相关变量
    cv::VideoCapture m_video_capture{};
    VideoInfo m_video_info{};
    int m_frame_step{};
    std::size_t m_image_node_num{};

    std::mutex m_mutex{};
    std::condition_variable m_frame_available{};
    std::condition_variable m_space_available{};  //检查缓冲节点是否有空位
    std::queue<VideoFrame> m_frame_queue{};

    //这两个标志的全部读写均由 m_mutex 保护，与队列状态一起判断
    bool m_stop_requested{false};
    bool m_finished{false};
    int m_seek_target{-1};
    int m_min_source_index{};
    std::uint64_t m_seek_version{};  //定位前后的解码结果必须属于同一次请求

    std::thread m_read_thread{};
};
