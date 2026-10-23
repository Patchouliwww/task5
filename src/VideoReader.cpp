#include <VideoReader.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>


VideoReader::VideoReader(
    const std::string& video_path,
    int frame_step,
    std::size_t image_node_num
)
    : m_frame_step{frame_step}
    , m_image_node_num{image_node_num}  //节点默认为3
{
    if (!m_video_capture.open(video_path))
    {
        throw std::runtime_error("无法打开视频：" + video_path + "\n");
    }

    //在线程启动前读取信息，之后只由后台线程操作视频读取对象
    m_video_info.width = static_cast<int>(m_video_capture.get(cv::CAP_PROP_FRAME_WIDTH));  //显式转换数据类型

    m_video_info.height = static_cast<int>(m_video_capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    m_video_info.fps = m_video_capture.get(cv::CAP_PROP_FPS);
    m_video_info.frame_count = static_cast<int>(m_video_capture.get(cv::CAP_PROP_FRAME_COUNT));

    if (m_video_info.fps <= 0 || m_video_info.frame_count <= 0)  //播放计时和滑条需要有效的视频信息
    {
        throw std::runtime_error("视频帧率或总帧数无效，无法建立播放进度\n");
    }

    //先完成打开和信息检查，启动读取线程
    m_read_thread = std::thread{&VideoReader::read_loop, this};
}


VideoReader::~VideoReader()
{
    //先结束线程，再由 m_video_capture 的析构释放资源，避免边读取边释放
    stop();
}


const VideoInfo& VideoReader::get_info() const
{
    return m_video_info;
}


bool VideoReader::get_frame(VideoFrame& output_frame)
{
    //等待有帧、自然读完或收到停止请求，谓词也能处理虚假唤醒
    std::unique_lock<std::mutex> lock{m_mutex};

    m_frame_available.wait(
        lock,
        [this] {
            return m_stop_requested || !m_frame_queue.empty() || m_finished;
        }
    );

    if (m_stop_requested)
    {
        return false;
    }

    if (!m_frame_queue.empty())
    {
        //先取出一帧，再解锁并通知读取线程，缩短占用共享锁的时间
        output_frame = std::move(m_frame_queue.front());

        m_frame_queue.pop();

        lock.unlock();

        m_space_available.notify_one();

        return true;
    }

    return false;
}


bool VideoReader::try_get_frame(VideoFrame& output_frame)
{
    std::unique_lock<std::mutex> lock{m_mutex};
    if (m_stop_requested || m_frame_queue.empty())
    {
        return false;
    }

    output_frame = std::move(m_frame_queue.front());
    m_frame_queue.pop();
    lock.unlock();
    m_space_available.notify_one();
    return true;
}


void VideoReader::seek(int source_index)
{
    {
        std::lock_guard<std::mutex> lock{m_mutex};
        m_frame_queue = {};
        m_seek_target = source_index;
        m_min_source_index = source_index;
        ++m_seek_version;
        m_finished = false;
    }
    m_space_available.notify_one();
}


void VideoReader::skip_to(int source_index)
{
    {
        std::lock_guard<std::mutex> lock{m_mutex};
        m_min_source_index = std::max(m_min_source_index, source_index);
        while (!m_frame_queue.empty() && m_frame_queue.front().source_index < m_min_source_index)
        {
            m_frame_queue.pop();
        }
    }
    m_space_available.notify_one();
}


void VideoReader::stop()
{
    {
        std::lock_guard<std::mutex> lock{m_mutex};

        m_stop_requested = true;
    }

    //生产者可能在等空位，消费者可能在等图像，取消时两边都要叫醒
    m_space_available.notify_all();

    m_frame_available.notify_all();

    if (m_read_thread.joinable())
    {
        m_read_thread.join();
    }
}


void VideoReader::read_loop()
{
    try
    {
        int next_source_frame_index{};  //标识该帧在原视频的编号

        while (true)
        {
            int seek_target{-1};
            std::uint64_t seek_version{};
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                //读到末尾后仍等待定位请求；缓存满时，定位和停止也能唤醒线程。
                m_space_available.wait(lock, [this]
                {
                    return m_stop_requested || m_seek_target >= 0
                        || (!m_finished && m_frame_queue.size() < m_image_node_num);
                });
                if (m_stop_requested)
                {
                    break;
                }
                seek_target = m_seek_target;
                m_seek_target = -1;
                seek_version = m_seek_version;
            }

            if (seek_target >= 0)
            {
                if (!m_video_capture.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(seek_target)))
                {
                    throw std::runtime_error("视频定位失败：" + std::to_string(seek_target));
                }
                next_source_frame_index = seek_target;
            }

            //顺序推进视频，跳过的位置不调用 retrieve；不反复按帧号跳转
            if (!m_video_capture.grab())
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                if (seek_version == m_seek_version)
                {
                    m_finished = true;
                }
                m_frame_available.notify_all();
                continue;
            }

            const int current_source_frame_index{next_source_frame_index};

            next_source_frame_index++;  //等同于const int current_source_frame_index{next_source_frame_index++};

            {
                std::lock_guard<std::mutex> lock{m_mutex};
                if (seek_version != m_seek_version || current_source_frame_index < m_min_source_index
                    || (seek_target < 0 && current_source_frame_index % m_frame_step != 0))
                {
                    continue;
                }
            }

            //获取需要的帧
            cv::Mat decoded_image{};  //用于解码得到的图像

            if (!m_video_capture.retrieve(decoded_image))  //检查读取是否成功
            {
                break;
            }

            //转移到更为全面的变量上
            VideoFrame frame_to_enqueue{};  //用于获取完整信息（包含图像和源帧号）

            frame_to_enqueue.image = decoded_image.clone();

            frame_to_enqueue.source_index = current_source_frame_index;

            //对类中定义的变量进行赋值，需要上锁
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                //解码期间可能发生定位或时间追赶，不能把已过期的帧重新放入队列。
                if (m_stop_requested || seek_version != m_seek_version
                    || current_source_frame_index < m_min_source_index)
                {
                    continue;
                }
                m_frame_queue.push(std::move(frame_to_enqueue));  //转移到可以被外界操作的变量，同时排到队尾
            }

            //唤醒get_frame
            m_frame_available.notify_one();
        }
    }
    catch (const std::exception& error)
    {
        //后台读取出错时打印原因，然后结束取帧
        std::cerr << error.what() << '\n';
    }


    {
        std::lock_guard<std::mutex> lock{m_mutex};

        m_finished = true;
    }

    m_frame_available.notify_all();
}
