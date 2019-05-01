#include <glog/logging.h>

#include "replay/rendering/depth_map_renderer.h"

#include "replay/sfm/reconstruction.h"

#include "replay/depth_map/depth_map.h"

namespace replay {
namespace {
static const std::string vertex_source =
    "#version 410\n"
    "uniform mat4 MVP;\n"
    "in vec3 vert;"
    "out vec3 xyz;"
    "void main()\n"
    "{\n"
    "    gl_Position = MVP * vec4(vert, 1.0);\n"
    "    xyz = vert;"
    "}\n";
static const std::string fragment_source =
    "#version 410\n"
    "out float color;\n"
    "uniform vec3 pos;\n"
    "in vec3 xyz;\n"
    "void main()\n"
    "{\n"
    "    color = distance(xyz,pos);\n"
    "}\n";
}  // namespace

DepthMapRenderer::DepthMapRenderer(std::shared_ptr<OpenGLContext> renderer)
    : renderer_(renderer), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize renderer first!";
}

bool DepthMapRenderer::Initialize() {
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    return false;
  }

  renderer_->HideWindow();
  is_initialized_ = true;
  renderer_->UseShader(shader_id_);
  return true;
}

void DepthMapRenderer::GetDepthMap(const Camera& camera,
                                   DepthMap* output_depth) {
  CHECK_NOTNULL(output_depth);
  CHECK(is_initialized_) << "Initialize reprojector first.";
  CHECK(renderer_->UseShader(shader_id_));
  CHECK(renderer_->SetViewpoint(camera));
  Eigen::Vector3f position = camera.GetPosition().cast<float>();
  renderer_->UploadShaderUniform(position, "pos");
  renderer_->RenderToImage(output_depth);
}

}  // namespace replay
