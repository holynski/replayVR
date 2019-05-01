#pragma once

#include <ceres/ceres.h>
#include <replay/rendering/opengl_context.h>
#include <opencv2/opencv.hpp>

namespace replay {

class LayerRefiner {
 public:
  LayerRefiner(const int width, const int height);

  bool AddImage(const cv::Mat3b& image, const cv::Mat2f& layer1_mapping,
                const cv::Mat2f& layer2_mapping);

  bool Optimize(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img,
                cv::Mat1f& alpha_mask);

 private:
  ceres::Problem problem_;
  std::vector<double> parameter_blocks_;
  int width_;
  int height_;

};

}  // namespace replay
