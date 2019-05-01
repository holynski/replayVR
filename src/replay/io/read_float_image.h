#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace replay {

cv::Mat1f ReadFloatImage(const std::string& filename);

}
