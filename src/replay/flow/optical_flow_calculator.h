#pragma once

#include <opencv2/opencv.hpp>

namespace replay {

class OpenGLContext;

enum OpticalFlowType {
  DIS,
  Simple,
  Farneback,
  TVL1,
  DeepFlow,
  SparseToDense,
  Greedy
};

class OpticalFlowCalculator {
 public:
  OpticalFlowCalculator(const OpticalFlowType& type,
                        std::shared_ptr<OpenGLContext> context = nullptr);

  cv::Mat2f ComputeFlow(const cv::Mat& base, const cv::Mat& target,
                        const cv::Mat2f& initialization = cv::Mat2f()) const;

 private:
  cv::Ptr<cv::DenseOpticalFlow> flow_;
  const OpticalFlowType& type_;
  std::shared_ptr<OpenGLContext> context_;
};

}  // namespace replay
