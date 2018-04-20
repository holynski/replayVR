#include "replay/rendering/vr180_renderer.h"
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/vr_context.h"
#include "replay/third_party/theia/sfm/types.h"
#include "replay/util/timer.h"
#include "replay/util/types.h"
#include "replay/vr_180/vr_180_video_reader.h"
#include <Eigen/Dense>
#include <openvr.h>

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

} // namespace

VR180Renderer::VR180Renderer(std::shared_ptr<VRContext> renderer)
    : renderer_(renderer), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize OpenGL renderer first!";
  clock_.Start();
}

namespace {} // namespace

bool VR180Renderer::Initialize(const std::string &spherical_video_filename) {
  renderer_->HideWindow();
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(ERROR) << "Couldn't compile shader!";
    return false;
  }

  renderer_->SetViewportSize(1000, 1000);

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

  cv::Mat3b image;

  CHECK(renderer_->UseShader(shader_id_));
  LOG(INFO) << "Uploading frames...";

  int total_frames = 0;
  Eigen::Matrix3f rotation;
  int index = 0;

  // TODO(holynski): Load metadata without decoding frames
  while (reader_.FetchOrientedFrame()) {
    rotation = reader_.GetFetchedOrientation();
    total_frames++;
    LOG(INFO) << "Added frame " << total_frames;
    frame_rotations_.emplace_back(rotation);
    frame_lookats_.push_back(-frame_rotations_[index].row(2));
    frame_upvecs_.push_back(frame_rotations_[index].row(1));
    frames_[index] = reader_.GetFetchedFrame();
    index++;
  }
  current_frame_ = 0;
  LOG(INFO) << "Loaded " << index << "/" << total_frames << " frames.";
  LOG(INFO) << "Done. Found " << index << " frames.";
  renderer_->ShowWindow();
  mesh_ids_.push_back(renderer_->UploadMesh(meshes_[0]));
  mesh_ids_.push_back(renderer_->UploadMesh(meshes_[1]));
  CHECK_GE(mesh_ids_[0], 0);
  CHECK_GE(mesh_ids_[1], 0);
  renderer_->UploadTexture(frames_[0], "frame");
  return true;
}

void VR180Renderer::Render() {
  CHECK(is_initialized_) << "Initialize renderer first.";

  CHECK(is_initialized_) << "Initialize renderer first.";
  CHECK(renderer_->UseShader(shader_id_));

  renderer_->UpdatePose();
  Eigen::Matrix3f hmd_rotation =
      renderer_->GetHMDPose().block(0, 0, 3, 3).transpose();
  
  // Start a timer from each render call to the next, so we don't play back the video too quickly.
  clock_.Stop();
  if (clock_.Count() < 30) {
	  clock_.Start();
    return;
  }
  clock_.Clear();
  clock_.Start();

  CHECK(renderer_->UseShader(shader_id_));

  renderer_->UpdateTexture(frames_[current_frame_], "frame");

  Eigen::Matrix4f mvp = Eigen::Matrix4f::Identity();
  mvp.block(0, 0, 3, 3) *=
      renderer_->GetHMDPose().block(0, 0, 3, 3).transpose() *
      frame_rotations_[current_frame_];

  Eigen::Matrix4f left_mvp, right_mvp;
  left_mvp = renderer_->GetProjectionMatrix(0) * mvp;
  right_mvp = renderer_->GetProjectionMatrix(1) * mvp;

  CHECK(renderer_->BindMesh(mesh_ids_[0]));
  renderer_->SetProjectionMatrix(left_mvp);
  renderer_->UploadShaderUniform(0, "right");
  renderer_->RenderEye(0);
  CHECK(renderer_->BindMesh(mesh_ids_[1]));
  renderer_->SetProjectionMatrix(right_mvp);
  renderer_->UploadShaderUniform(1, "right");
  renderer_->RenderEye(1);
  current_frame_++;
  if (current_frame_ >= frames_.size()) {
    current_frame_ = 0;
  }

}

} // namespace replay
