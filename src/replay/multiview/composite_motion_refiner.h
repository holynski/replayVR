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
  const int width_;
  const int height_;
  std::vector<Eigen::Vector2i> index_to_coord_;
  cv::Mat1i coord_to_index_;
  std::vector<double> parameters_;
};

}  // namespace replay
