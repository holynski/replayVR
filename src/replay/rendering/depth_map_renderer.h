#pragma once

#include "replay/camera/camera.h"

#include "replay/rendering/opengl_context.h"

namespace replay {

class DepthMap;
class Mesh;

class DepthMapRenderer {
 public:
  // The depth map rasterizer takes in a mesh. The depths maps are rasterized
  // from a given input camera's viewpoint using OpenGL.
  explicit DepthMapRenderer(std::shared_ptr<OpenGLContext> renderer);

  // This must be called prior to calling GetDepthMap.
  bool Initialize();

  // Computes a depth map for the given camera by raycasting onto the mesh.
  void GetDepthMap(const Camera& camera, DepthMap* output_depth);

 private:
  std::shared_ptr<OpenGLContext> renderer_;
  int shader_id_;
  bool is_initialized_;
};

}  // namespace replay

