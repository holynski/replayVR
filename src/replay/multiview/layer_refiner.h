#pragma once

#include <ceres/ceres.h>
#include <replay/rendering/opengl_context.h>
#include <opencv2/opencv.hpp>

namespace replay {

class LayerRefiner {
 public:
  LayerRefiner(const int width, const int height);

  bool AddImage(const cv::Mat3b& image, const cv::Mat2f& flow_to_layer1,
                const cv::Mat2f& flow_to_layer2,
                const cv::Mat1b& valid_pixels = cv::Mat1b());

  bool Optimize(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img, cv::Mat1f& alpha,
                const int num_iterations);

 private:
  const int width_;
  const int height_;
  std::vector<Eigen::Vector2i> index_to_coord_;
  cv::Mat1i coord_to_index_;
  std::vector<double> parameters_;
  std::vector<double> observations_;
  int num_images_;
  ceres::Problem problem_;
};

}  // namespace replay
