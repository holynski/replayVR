#pragma once

#include <opencv2/opencv.hpp>
#include <replay/flow/optical_flow_calculator.h>

namespace replay {

class OpticalFlowAligner {
 public:
  OpticalFlowAligner(const OpticalFlowType& type);
  cv::Mat FlowWarp(const cv::Mat& input, const cv::Mat2f& flow) const;
  cv::Mat Align(const cv::Mat& base, const cv::Mat& target) const;

 private:
  OpticalFlowCalculator flow_;
};

}  // namespace replay
