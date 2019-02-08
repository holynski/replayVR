#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "replay/camera/camera.h"
#include "replay/image/sum_absolute_difference.h"
#include "replay/rendering/image_reprojector.h"

namespace replay {

class PlaneSweep {
 public:
  PlaneSweep(std::shared_ptr<OpenGLContext> context);

  cv::Mat1f GetCost(const Camera& camera1, const Camera& camera2,
                    const cv::Mat3b& image1, const cv::Mat1b& valid_pixels1,
                    const cv::Mat3b& image2, const cv::Mat1b& valid_pixels2,
                    const Eigen::Vector4f& plane);

  // Searches for a single dominant plane to represent a scene
  std::unordered_map<float, cv::Mat1f> Sweep(const Camera& camera1, const Camera& camera2,
                    const cv::Mat3b& image1, const cv::Mat1b& valid_pixels1,
                    const cv::Mat3b& image2, const cv::Mat1b& valid_pixels2,
                    const float min_depth, const float max_depth, const int num_steps);

 private:
  std::shared_ptr<OpenGLContext> context_;
  SumAbsoluteDifference<cv::Vec3b> sad_;
  ImageReprojector reprojector_;
};

}  // namespace replay
