#pragma once

#include <opencv2/opencv.hpp>

namespace replay {

class OpenGLContext;

class GreedyFlow : public cv::DenseOpticalFlow {
 public:
  static cv::Ptr<GreedyFlow> Create();

  GreedyFlow(std::shared_ptr<OpenGLContext> context, const int window_size = 5);

  void SetWindowSize(const int window_size);

  void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow);

  void collectGarbage();

 private:
  std::shared_ptr<OpenGLContext> context_;
  int window_size_;
  int shader_id_;
};

}  // namespace replay
