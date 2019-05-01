#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace replay {

bool WriteFloatImage(const std::string& filename, const cv::Mat1f& image);

}
