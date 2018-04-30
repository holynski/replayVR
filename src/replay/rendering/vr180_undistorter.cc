#include "replay/rendering/vr180_undistorter.h"
#include <openvr.h>
#include <Eigen/Dense>
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/vr_context.h"
#include "replay/third_party/theia/sfm/types.h"
#include "replay/util/types.h"
#include "replay/vr_180/vr_180_video_reader.h"

namespace replay {
namespace {
static const std::string vertex_source =
    "#version 410\n"
    "uniform mat4 MVP;\n"
    "in vec3 vert;"
    "in vec2 uv;"
    "out vec3 pos;"
    "out vec2 frag_uv;"
    "uniform int right;"
    "void main()\n"
    "{\n"
    "    pos = vert;"
    "    frag_uv = uv;"
    "         gl_Position = MVP * vec4(vert, 1.0);\n"
    "    if (right == 0) {"
    "         frag_uv.x *= 0.5;"
    "    }"
    "    else {"
    "         frag_uv.x *= 0.5;"
    "         frag_uv.x += 0.5;"
    "    }"
    "}\n";
static const std::string fragment_source =
    "#version 410\n"
    "out vec3 color;\n"
    "in vec3 pos;"
    "in vec2 frag_uv;"
    "uniform sampler2D frame;"
    "void main()\n"
    "{\n"
    "    color = texture(frame, vec2(frag_uv)).rgb;"
    "}\n";

}  // namespace

VR180Undistorter::VR180Undistorter(std::shared_ptr<OpenGLContext> renderer)
    : renderer_(renderer), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize OpenGL renderer first!";
}

namespace {}  // namespace

bool VR180Undistorter::Initialize(const std::string& spherical_video_filename, const Camera& camera) {
  camera_ = camera;
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(ERROR) << "Couldn't compile shader!";
    return false;
  }

  renderer_->HideWindow();
  is_initialized_ = true;
  if (!reader_.Open(spherical_video_filename)) {
    LOG(ERROR) << "Couldn't open spherical video file: "
               << spherical_video_filename;
    return false;
  }

  meshes_ = reader_.GetMeshes();
  if (meshes_.size() != 2) {
    LOG(ERROR) << "No left/right projection meshes.";
    return false;
  }

  CHECK(renderer_->UseShader(shader_id_));

  mesh_ids_.push_back(renderer_->UploadMesh(meshes_[0]));
  mesh_ids_.push_back(renderer_->UploadMesh(meshes_[1]));
  CHECK_GE(mesh_ids_[0], 0);
  CHECK_GE(mesh_ids_[1], 0);

  return true;
}

bool VR180Undistorter::UndistortFrame(cv::Mat3b* left, cv::Mat3b* right) {
  CHECK(is_initialized_) << "Initialize renderer first.";

  CHECK(renderer_->UseShader(shader_id_));

  if (!reader_.FetchOrientedFrame()) {
    return false;
  }

  renderer_->UploadTexture(reader_.GetFetchedFrame(), "frame");

  CHECK(renderer_->BindMesh(mesh_ids_[0]));
  renderer_->SetViewpoint(camera_);
  renderer_->UploadShaderUniform(0, "right");
  renderer_->RenderToImage(left);
  CHECK(renderer_->BindMesh(mesh_ids_[1]));
  renderer_->SetViewpoint(camera_);
  renderer_->UploadShaderUniform(1, "right");
  renderer_->RenderToImage(right);
  return true;
}

}  // namespace replay
