#pragma once

#include <opencv2/opencv.hpp>

namespace replay {

enum class Colormap { Jet };

cv::Mat3b FloatToColor(const cv::Mat1f& image,
                       const Colormap cm = Colormap::Jet);

cv::Mat3b FloatToColor(const cv::Mat1f& image, const float min_scale,
                       const float max_scale,
                       const Colormap cm = Colormap::Jet);

}  // namespace replay
