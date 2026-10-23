#include <Vision.h>

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>


//构造函数
ImagePreprocessor::ImagePreprocessor(const PreprocessParams& params)
    : m_params{params}  //接收后用于初始化成员
{
}


//图像预处理
cv::Mat ImagePreprocessor::preprocess(const cv::Mat& bgr_image) const
{
    //高斯模糊，用以平滑边缘
    cv::Mat smoothed_image{};

    cv::GaussianBlur(
        bgr_image, 
        smoothed_image, 
        m_params.gaussian_kernel, 
        m_params.gaussian_sigma
    );

    //将图像从BGR格式转换为HSV格式
    cv::Mat hsv_image{};

    cv::cvtColor(smoothed_image, hsv_image, cv::COLOR_BGR2HSV);

    //筛选出红蓝并二值化
    cv::Mat low_mask{}, high_mask{}, binary_image{};

    cv::inRange(
        hsv_image,
        m_params.low_range.lower_bound,
        m_params.low_range.upper_bound,
        low_mask
    );

    cv::inRange(
        hsv_image,
        m_params.high_range.lower_bound,
        m_params.high_range.upper_bound,
        high_mask
    );

    cv::bitwise_or(low_mask, high_mask, binary_image);

    //闭运算：先膨胀，再腐蚀
    const cv::Mat kernel{cv::getStructuringElement(cv::MORPH_RECT, m_params.closing_kernel)};  //生成3*3的矩形

    cv::morphologyEx(binary_image, binary_image, cv::MORPH_CLOSE, kernel);

    return binary_image;
}


ArmorDetector::ArmorDetector(
    const LightBarFilterParams& filter_params,
    const ArmorPairParams& pair_params
)
    : m_filter_params{filter_params},
      m_pair_params{pair_params}
{
}



std::vector<LightBar> ArmorDetector::find_light_bars(const cv::Mat& binary_image) const
{
    if (binary_image.empty())
    {
        return {};
    }

    cv::Mat labels{}, stats{}, centroids{};
    const int region_count{cv::connectedComponentsWithStats(binary_image, labels, stats, centroids, 8)};

    std::vector<LightBar> light_bars{};  

    //依次获取灯条的参数
    for (int label{1}; label < region_count; ++label)
    {
        //使用统计的白色像素数量，先筛选面积，再测量方向和长宽。
        const int area{stats.at<int>(label, cv::CC_STAT_AREA)};

        if (area < m_filter_params.min_area || area > m_filter_params.max_area)
        {
            continue;
        }

        //获取接下来处理图像的范围
        const cv::Rect bounds{
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT)
        };
        
        const cv::Mat region_mask{labels(bounds) == label};  //保证bounds框定的灯条皆为同一条

        //获取参数
        LightBar light_bar{};
        if (!measure_light_bar(region_mask, bounds.tl(), light_bar))
        {
            continue;
        }

        //根据上一段获得的参数进行筛选
        if (passes_filter(light_bar))
        {
            light_bars.push_back(light_bar);
        }
    }

    return light_bars;
}


bool ArmorDetector::measure_light_bar(
    const cv::Mat& region_mask,
    const cv::Point& offset,
    LightBar& light_bar
)
{
    const cv::Moments moments{cv::moments(region_mask, true)};  //计算白色像素的图像矩

    const double spread{moments.mu20 + moments.mu02};  //表示总体分散程度

    const double direction_strength{
        std::hypot(
            moments.mu20 - moments.mu02, 
            2.0 * moments.mu11
        )
    };  //计算是否有一个明显的伸展方向

    if (direction_strength <= 1e-6 * spread)  //排除无主轴的图形
    {
        return false;
    }

    //计算中转参数
    const cv::Point2d center{
        moments.m10 / moments.m00, 
        moments.m01 / moments.m00
    };  //计算质心

    const double theta{
        0.5 * std::atan2(
            2.0 * moments.mu11, 
            moments.mu20 - moments.mu02
        )
    };  //计算主轴方向

    const cv::Point2d axis{std::cos(theta), std::sin(theta)};  //与主轴同向的单位方向向量

    const cv::Point2d normal{-axis.y, axis.x};  //与主轴垂直的单位方向向量

    //获取最大长宽
    std::vector<cv::Point> pixels{};
    cv::findNonZero(region_mask, pixels);
    
    double half_length{}, half_width{};

    for (const cv::Point& pixel : pixels)
    {
        const cv::Point2d relative{
            pixel.x - center.x, 
            pixel.y - center.y
        };  //质心到白点的向量

        half_length = std::max(
            half_length, 
            std::abs(relative.dot(axis))
        );

        half_width = std::max(
            half_width, 
            std::abs(relative.dot(normal))
        );
    }

    //框以质心为中心，取两侧较远距离，保证两侧白点都能包含；度量仍以像素中心为准。
    if (half_width <= 0.0)
    {
        return false;
    }

    //theta在[-90°,90°]内；加90°改成相对竖直的角度，再折回[-90°,90°)。
    float angle{static_cast<float>(theta * 180.0 / CV_PI + 90.0)};

    if (angle >= 90.0F)
    {
        angle -= 180.0F;
    }

    //转移数据
    light_bar.center = cv::Point2f{
        static_cast<float>(center.x + offset.x), 
        static_cast<float>(center.y + offset.y)
    };
    light_bar.length = static_cast<float>(2.0 * half_length);
    light_bar.width = static_cast<float>(2.0 * half_width);
    light_bar.angle = angle;

    return true;
}

//主轴法已在测量前筛过像素面积，这里只检查长度、长宽比和倾角。
bool ArmorDetector::passes_filter(const LightBar& light_bar) const
{
    const float aspect_ratio{light_bar.length / light_bar.width};

    return light_bar.length >= m_filter_params.min_length
        && aspect_ratio >= m_filter_params.min_aspect_ratio
        && aspect_ratio <= m_filter_params.max_aspect_ratio
        && std::abs(light_bar.angle) <= m_filter_params.max_abs_angle;
}



std::vector<Armor> ArmorDetector::pair_light_bars(const std::vector<LightBar>& light_bars) const
{
    struct PairCandidate
    {
        std::size_t left_index{};
        std::size_t right_index{};
        float score{};
    };

    std::vector<PairCandidate> candidates{};

    for (std::size_t first{0}; first < light_bars.size(); ++first)
    {
        for (std::size_t second{first + 1}; second < light_bars.size(); ++second)
        {
            //判断成对灯条相对位置以区分左右
            const bool first_is_left{light_bars[first].center.x <= light_bars[second].center.x};

            const std::size_t left_index{first_is_left ? first : second};

            const std::size_t right_index{first_is_left ? second : first};
            
            float score{};

            if (passes_pair_filter(light_bars[left_index], light_bars[right_index], score))
            {
                candidates.push_back({left_index, right_index, score});
            }
        }
    }

    //比较全部合格组合，同分时保持原顺序
    std::stable_sort(candidates.begin(), candidates.end(), [](const PairCandidate& first, const PairCandidate& second)
    {
        return first.score < second.score;  //返回分数小的
    });

    std::vector<bool> used_light_bars(light_bars.size(), false);
    std::vector<Armor> armors{};

    for (const PairCandidate& candidate : candidates)
    {
        if (used_light_bars[candidate.left_index] || used_light_bars[candidate.right_index])
        {
            continue;
        }

        //生成装甲板并标记为已使用
        armors.push_back(make_armor(light_bars[candidate.left_index], light_bars[candidate.right_index]));
        used_light_bars[candidate.left_index] = true;
        used_light_bars[candidate.right_index] = true;
    }

    return armors;
}



//进行初步的几何筛选并生成分数
bool ArmorDetector::passes_pair_filter(
    const LightBar& left,
    const LightBar& right,
    float& score
) const
{
    const float length_difference_ratio{
        std::abs(left.length - right.length) / std::max(left.length, right.length)
    };
    const float angle_difference{std::abs(left.angle - right.angle)};

    //判断相对长度差
    if (length_difference_ratio > m_pair_params.max_length_difference_ratio
        || angle_difference > m_pair_params.max_angle_difference)
    {
        return false;
    }

    const float average_length{(left.length + right.length) / 2.0F};  //获取平均长度

    const double average_angle{(left.angle + right.angle) / 2.0 * CV_PI / 180.0};  //获取平均倾角并转换为弧度制

    const cv::Point2f vertical_direction{
        static_cast<float>(std::sin(average_angle)),
        static_cast<float>(-std::cos(average_angle))
    };  //获取沿灯条方向向量

    const cv::Point2f horizontal_direction{
        static_cast<float>(std::cos(average_angle)),
        static_cast<float>(std::sin(average_angle))
    };  //获取垂直于灯条方向向量

    const cv::Point2f center_offset{right.center - left.center};  //获取中心位移向量

    //沿共同方向测错位、沿垂直方向测间距，再除以平均长度减小远近的影响
    const float vertical_offset_ratio{std::abs(center_offset.dot(vertical_direction)) / average_length};
    const float spacing_ratio{std::abs(center_offset.dot(horizontal_direction)) / average_length};

    if (vertical_offset_ratio > m_pair_params.max_vertical_offset_ratio
        || spacing_ratio < m_pair_params.min_spacing_ratio
        || spacing_ratio > m_pair_params.max_spacing_ratio
    )
    {
        return false;
    }

    //各项先除以对应上限再乘权重，分数越小越优先；零上限已由筛选要求该项为零
    const auto normalized_score = [](float value, float maximum)
    {
        return maximum > 0.0F ? value / maximum : 0.0F;
    };

    score = m_pair_params.angle_weight
              * normalized_score(angle_difference, m_pair_params.max_angle_difference)
          + m_pair_params.length_weight
              * normalized_score(length_difference_ratio, m_pair_params.max_length_difference_ratio)
          + m_pair_params.vertical_offset_weight
              * normalized_score(vertical_offset_ratio, m_pair_params.max_vertical_offset_ratio);

    return true;
}


//生成装甲板
Armor ArmorDetector::make_armor(const LightBar& left, const LightBar& right)
{
    const double left_angle{left.angle * CV_PI / 180.0};
    const double right_angle{right.angle * CV_PI / 180.0};

    //计算水平和数值位移
    const cv::Point2f left_half_length{
        static_cast<float>(std::sin(left_angle) * left.length / 2.0),
        static_cast<float>(-std::cos(left_angle) * left.length / 2.0)
    };
    const cv::Point2f right_half_length{
        static_cast<float>(std::sin(right_angle) * right.length / 2.0),
        static_cast<float>(-std::cos(right_angle) * right.length / 2.0)
    };

    Armor armor{};

    armor.corners = {
        left.center + left_half_length,
        right.center + right_half_length,
        right.center - right_half_length,
        left.center - left_half_length
    };

    return armor;
}



