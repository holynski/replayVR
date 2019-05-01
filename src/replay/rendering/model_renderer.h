#pragma once

#include "replay/camera/camera.h"
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/opengl_context.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"

namespace replay {

class ModelRenderer {
 public:
  explicit ModelRenderer(std::shared_ptr<OpenGLContext> renderer);

  bool RenderView(const Camera& viewpoint, cv::Mat* output);
  // For orthographic projection
  bool RenderView(const Eigen::Matrix4f& projection,
                  const Eigen::Vector2i& image_size, cv::Mat* output);
  bool Initialize();

 private:
  std::shared_ptr<OpenGLContext> renderer_;
  int shader_id_;
  bool is_initialized_;
};
}  // namespace replay

