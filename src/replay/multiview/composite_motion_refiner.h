#pragma once

#include <ceres/ceres.h>
#include <replay/rendering/opengl_context.h>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

namespace replay {

class CompositeMotionRefiner {
 public:
  CompositeMotionRefiner(const int width, const int height);

  bool Optimize(const cv::Mat3b& layer1_img, const cv::Mat3b& layer2_img,
                const cv::Mat1f& alpha_img, const cv::Mat3b& composite,
                cv::Mat2f& layer1, cv::Mat2f& layer2, const int num_iterations);

 private:
  bool GradientDescent(const cv::Mat3b& layer1_img, const cv::Mat3b& layer2_img,
                       const cv::Mat1f& alpha_img, const cv::Mat3b& composite,
                       cv::Mat2f& layer1, cv::Mat2f& layer2);
  const int width_;
  const int height_;
  std::vector<Eigen::Triplet<double>> triplets_;
  std::vector<double> b_;
  double current_row_;
  std::vector<Eigen::Vector2i> vars_to_pixels_;
  cv::Mat1i pixels_to_vars_;
};

}  // namespace replay
