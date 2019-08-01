#pragma once

#include <opencv2/opencv.hpp>

namespace replay {

namespace OpticalFlow {

// Takes an optical flow field and scales it by a given amount. The flow values
// are also scaled proportionally.
cv::Mat2f Scale(const cv::Mat2f& flow, const float scale);

cv::Mat InverseWarp(const cv::Mat& src, const cv::Mat2f& flow_dest_to_source);

}  // namespace OpticalFlow

}  // namespace replay
