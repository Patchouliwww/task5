#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>


//一段HSV筛选范围，两个边界均包含在内。
struct HsvRange
{
    cv::Scalar lower_bound{};
    cv::Scalar upper_bound{};
};


//预处理参数
struct PreprocessParams
{
    //两组HSV参数，筛选出红和蓝
    HsvRange low_range{cv::Scalar{0, 20, 130}, cv::Scalar{10, 255, 255}};
    HsvRange high_range{cv::Scalar{90, 20, 130}, cv::Scalar{179, 255, 255}};

    cv::Size gaussian_kernel{5, 5};  //高斯核尺寸
    double gaussian_sigma{0.0};  //0表示根据核尺寸计算，当前X/Y方向采用相同设置
    cv::Size closing_kernel{3, 3};  //矩形闭运算核尺寸
};


class ImagePreprocessor
{
public:
    explicit ImagePreprocessor(
        const PreprocessParams& params = PreprocessParams{}
    );  //不传参数的话就使用默认对象

    cv::Mat preprocess(const cv::Mat& bgr_image) const;  //接收BGR图，经滤波、双段HSV及闭运算后返回二值图
private:
    PreprocessParams m_params{};  //预处理和参数记录共用同一组设置
};



//单灯条筛选与后续识别共用的几何信息
struct LightBar
{
    cv::Point2f center{};  //二维中心坐标

    //长度沿主轴、宽度沿垂直方向测量，均以像素计
    float length{};
    float width{};

    //长边相对竖直方向的倾角：上端向右为正，范围为 [-90, 90) 度
    float angle{};
};


struct Armor
{
    //左右灯条中心线的端点，依次为左上、右上、右下、左下
    std::array<cv::Point2f, 4> corners{};
};


//单灯条筛选参数
struct LightBarFilterParams
{
    //面积表示每个独立白色区域的像素数量。
    double min_area{5.0};
    double max_area{250.0};

    float min_length{4.0F};  //最小长边，单位为像素
    float min_aspect_ratio{1.0F};  //最小长宽比
    float max_aspect_ratio{8.0F};  //最大长宽比
    float max_abs_angle{30.0F};  //相对竖直方向的最大倾角绝对值，单位为度
};


//配对参数设置
struct ArmorPairParams
{
    float max_length_difference_ratio{0.45F};  //最大相对长度差，公式为差值除以最大值
    float max_angle_difference{17.0F};  //最大方向差，单位为度
    float max_vertical_offset_ratio{0.8F};  //最大纵向错位/平均灯条长度
    float min_spacing_ratio{1.5F};  //最小横向间距/平均灯条长度
    float max_spacing_ratio{5.8F};  //最大横向间距/平均灯条长度

    float angle_weight{5.0F};  //方向差评分权重，越大越优先选择方向接近的组合
    float length_weight{1.0F};  //相对长度差评分权重，越大越优先选择长度接近的组合
    float vertical_offset_weight{3.0F};  //纵向错位评分权重，越大越优先选择纵向对齐的组合
};


//检测装甲板的类
class ArmorDetector
{
public:
    explicit ArmorDetector(
        const LightBarFilterParams& filter_params = LightBarFilterParams{},
        const ArmorPairParams& pair_params = ArmorPairParams{}
    );

    //按白色区域主轴测量并筛选单灯条，数量可以为零；不负责配对或显示
    std::vector<LightBar> find_light_bars(const cv::Mat& binary_image) const;

    //输入已有候选灯条；对合格组合评分择优，每根候选灯条最多参与一个结果
    std::vector<Armor> pair_light_bars(const std::vector<LightBar>& light_bars) const;

private:
    //根据实际白点的图像矩求质心和主轴，再沿主轴及垂直方向测量长宽。
    static bool measure_light_bar(const cv::Mat& region_mask, const cv::Point& offset,
                                  LightBar& light_bar);

    bool passes_filter(const LightBar& light_bar) const;

    bool passes_pair_filter(const LightBar& left, const LightBar& right, float& score) const;

    static Armor make_armor(const LightBar& left, const LightBar& right);

    LightBarFilterParams m_filter_params{};
    ArmorPairParams m_pair_params{};
};



