#pragma once

#include "replay/depth_map/depth_map.h"
#include "replay/geometry/mesh.h"
#include "replay/rendering/opengl_context.h"
#include "replay/camera/camera.h"
#include "replay/sfm/reconstruction.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"

namespace replay {

class ImageBasedRenderer {
 public:
  explicit ImageBasedRenderer(std::shared_ptr<OpenGLContext> renderer);

  bool RenderView(const Camera& viewpoint, cv::Mat* output, int id);

  bool Initialize(const Reconstruction& reconstruction,
                  const ImageCache& cache,
                  const Mesh& mesh);

 private:
  std::shared_ptr<OpenGLContext> renderer_;
  int shader_id_;
  bool is_initialized_;
  int mesh_id_;
};
}  // namespace mvsa
