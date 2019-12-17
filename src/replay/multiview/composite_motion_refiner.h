#pragma once

#include <ceres/ceres.h>
#include <replay/rendering/opengl_context.h>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

namespace replay {

class CompositeMotionRefiner {
 public:
  static bool Optimize(const cv::Mat3b& layer1_img, const cv::Mat3b& layer2_img,
                       const cv::Mat1f& alpha_img, const cv::Mat3b& composite,
                       cv::Mat2f& layer1, cv::Mat2f& layer2,
                       const int num_iterations);

 private:
};

}  // namespace replay
