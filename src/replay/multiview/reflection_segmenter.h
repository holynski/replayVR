#pragma once

#include <GCoptimization.h>
#include <replay/flow/flow_from_reprojection.h>
#include <replay/image/fuzzy_difference.h>
#include <replay/flow/greedy_flow.h>
#include <replay/rendering/opengl_context.h>
#include <opencv2/opencv.hpp>

namespace replay {

class ReflectionSegmenter {
 public:
  ReflectionSegmenter(std::shared_ptr<OpenGLContext> context,
                      const Camera& reference_viewpoint,
                      const cv::Mat3b& layer1_img, const cv::Mat3b& layer2_img,
                      const Mesh& layer_1, const Mesh& layer_2);

  bool AddImage(const cv::Mat3b& image, const Camera& camera);

  bool Optimize(cv::Mat1b& mask,
                const cv::Mat1b& candidate_edges = cv::Mat1b());

 private:
  std::shared_ptr<OpenGLContext> context_;
  replay::FuzzyMinDifference<cv::Vec3b> min_difference_;
  replay::ImageReprojector image_reprojector_;
  FlowFromReprojection flow_calculator_;
  const int width_;
  const int height_;
  const Camera& reference_viewpoint_;
  const cv::Mat3b& layer1_img_;
  const cv::Mat3b& layer2_img_;
  const int layer_1_mesh_id_;
  const int layer_2_mesh_id_;

  cv::Mat1d reflection_cost_;
  cv::Mat1d diffuse_cost_;
  cv::Mat1d cost_count_;
};

}  // namespace replay

