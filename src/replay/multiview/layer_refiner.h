#pragma once

#include <ceres/ceres.h>
#include <replay/flow/flow_from_reprojection.h>
#include <replay/rendering/opengl_context.h>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

namespace replay {

class LayerRefiner {
 public:
  LayerRefiner(const int width, const int height);

  bool AddImage(const cv::Mat3b& image, const cv::Mat2f& flow_layer1,
                const cv::Mat2f& flow_layer2, const cv::Mat1b& mask);

  bool Optimize(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img,
                const int num_iterations);

 private:
  bool GradientDescent(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img);
  const int width_;
  const int height_;
  std::vector<Eigen::Triplet<double>> triplets_;
  std::vector<double> b_;
  double current_row_;
  std::vector<Eigen::Vector2i> vars_to_pixels_;
  cv::Mat1i pixels_to_vars_;
};

}  // namespace replay
