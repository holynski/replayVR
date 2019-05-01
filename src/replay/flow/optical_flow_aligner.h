#pragma once

#include <replay/flow/optical_flow_calculator.h>
#include <opencv2/opencv.hpp>

namespace replay {

class OpticalFlowAligner {
 public:
  OpticalFlowAligner(const OpticalFlowType& type,
                     std::shared_ptr<OpenGLContext> context = nullptr);
  static cv::Mat InverseWarp(const cv::Mat& src,
                             const cv::Mat2f& flow_dest_to_source);
  cv::Mat Align(const cv::Mat& base, const cv::Mat& target) const;

 private:
  OpticalFlowCalculator flow_;
};

}  // namespace replay
