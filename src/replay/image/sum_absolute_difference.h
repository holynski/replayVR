#pragma once

#include <glog/logging.h>
#include <replay/rendering/opengl_context.h>
#include <opencv2/opencv.hpp>

namespace replay {

template <typename T>
class SumAbsoluteDifference {
 public:
  SumAbsoluteDifference(std::shared_ptr<OpenGLContext> renderer);
  cv::Mat1f GetDifference(const cv::Mat_<T>& image1, const cv::Mat_<T>& image2,
                          const cv::Mat1b& mask1, const cv::Mat1b& mask2,
                          const int window_size = 10);

  std::shared_ptr<OpenGLContext> renderer_;
  int shader_id_;
  bool is_initialized_;
};

}  // namespace replay

#include "replay/image/sum_absolute_difference_impl.h"
