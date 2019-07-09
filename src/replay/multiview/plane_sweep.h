#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "replay/camera/camera.h"
#include "replay/image/sum_absolute_difference.h"
#include "replay/rendering/image_reprojector.h"

// TODO(holynski): Refactor the 2-view to a separate class

namespace replay {

struct PlaneSweepResult {
  std::vector<float> depths;
  std::unordered_map<float, cv::Mat1f> cost_volume;
  std::unordered_map<float, cv::Mat1f> num_samples;
  std::unordered_map<float, cv::Mat3b> mean_images;
  std::unordered_map<float, cv::Mat3b> max_images;
  std::unordered_map<float, cv::Mat3b> min_images;
  std::unordered_map<float, int> mesh_ids;
  std::unordered_map<float, Mesh> meshes;
};

class PlaneSweep {
 public:
  PlaneSweep(std::shared_ptr<OpenGLContext> context,
             const std::string& cache_directory = "");

  cv::Mat1f GetCost(const Camera& camera1, const Camera& camera2,
                    const cv::Mat3b& image1, const cv::Mat1b& valid_pixels1,
                    const cv::Mat3b& image2, const cv::Mat1b& valid_pixels2,
                    const Eigen::Vector4f& plane);

  // Searches for a single dominant plane to represent a scene
  std::unordered_map<float, cv::Mat1f> Sweep(
      const Camera& camera1, const Camera& camera2, const cv::Mat3b& image1,
      const cv::Mat1b& valid_pixels1, const cv::Mat3b& image2,
      const cv::Mat1b& valid_pixels2, const float min_depth,
      const float max_depth, const int num_steps);

  void AddView(const Camera& camera, const cv::Mat3b& image,
               const cv::Mat1b& valid_pixels);
  std::vector<cv::Mat3b> ProjectAllImagesToPlane(const int mesh_id,
                                                 const Camera& camera);
  PlaneSweepResult Sweep(const Camera& viewpoint, const float min_depth,
                         const float max_depth, const int num_steps);

 private:
  int CreatePlaneMesh(const Eigen::Vector4f& plane);

  std::shared_ptr<OpenGLContext> context_;
  SumAbsoluteDifference<cv::Vec3b> sad_;
  ImageReprojector reprojector_;
  const std::string& cache_directory_;

  std::vector<const Camera*> cameras_;
  std::vector<cv::Mat3b> images_;
  std::vector<cv::Mat1b> masks_;
};

}  // namespace replay
