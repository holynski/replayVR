#pragma once

#include "replay/camera/camera.h"
#include "replay/depth_map/depth_map.h"
#include "replay/rendering/depth_map_renderer.h"
#include "replay/rendering/opengl_context.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"

namespace replay {

class VarianceEstimator {
 public:
  explicit VarianceEstimator(std::shared_ptr<OpenGLContext> renderer);

  bool GetVariance(const Reconstruction& reconstruction,
                       const ImageCache& cache, const Camera& viewpoint,
                       cv::Mat* output);

  bool Initialize();

 private:
  const std::shared_ptr<OpenGLContext> renderer_;
  DepthMapRenderer depth_renderer_;
  int shader_id_;
  bool is_initialized_;
  std::unordered_map<int, DepthMap> depth_maps_;
};
}  // namespace replay
