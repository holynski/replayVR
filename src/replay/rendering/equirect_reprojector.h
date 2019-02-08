#pragma once

#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/opengl_context.h"
#include "replay/camera/camera.h"
#include "replay/sfm/reconstruction.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"

namespace replay {

class EquirectReprojector {
 public:
  explicit EquirectReprojector(std::shared_ptr<OpenGLContext> renderer);

  bool ProjectToEquirect(const cv::Mat& image, const Camera& camera, cv::Mat* output);
  bool ProjectFromEquirect(const cv::Mat& image, const Camera& camera, cv::Mat* output);

 private:
  std::shared_ptr<OpenGLContext> context_;
  int shader_id_to_;
  int shader_id_from_;
};
}  // namespace mvsa


