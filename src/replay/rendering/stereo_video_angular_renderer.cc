#include "replay/rendering/stereo_video_angular_renderer.h"
#include <openvr.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/vr_context.h"
#include "replay/third_party/theia/sfm/types.h"
#include "replay/util/timer.h"
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
    "uniform sampler2D image;"
    "void main()\n"
    "{\n"
    "    color = texture(image, frag_uv).rgb;"
    "}\n";
}  // namespace

StereoVideoAngularRenderer::StereoVideoAngularRenderer(
    std::shared_ptr<VRContext> renderer)
    : renderer_(renderer), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize OpenGL renderer first!";
}

namespace {}  // namespace

bool StereoVideoAngularRenderer::Initialize(
    const std::string &spherical_video_filename) {
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

  mesh_ids_.push_back(renderer_->UploadMesh(meshes_[0]));
  mesh_ids_.push_back(renderer_->UploadMesh(meshes_[1]));
  CHECK_GE(mesh_ids_[0], 0);
  CHECK_GE(mesh_ids_[1], 0);

  // TODO(holynski): Load metadata without decoding frames
  while (reader_.FetchOrientedFrame()) {
    rotation = reader_.GetFetchedOrientation();
    total_frames++;

    if (frame_lookats_.size() > 0 &&
        frame_lookats_[frame_lookats_.size() - 1].dot(-rotation.row(2)) >
            cos(angular_resolution_ * M_PI / 180.0)) {
      LOG(INFO) << "Skipped frame " << total_frames;
      continue;
    }
    LOG(INFO) << "Added frame " << total_frames;
    frame_rotations_.emplace_back(rotation);
    frame_lookats_.push_back(-frame_rotations_[index].row(2));
    frame_upvecs_.push_back(frame_rotations_[index].row(1));
    frames_[index] = reader_.GetFetchedFrame();
    index++;
  }
  current_frame_ = frame_lookats_.size() / 2;
  LOG(INFO) << "Loaded " << index << "/" << total_frames << " frames.";
  LOG(INFO) << "Done. Found " << index << " frames.";
  renderer_->ShowWindow();

  renderer_->UploadTexture(frames_[0], "image");
  return true;
}

void StereoVideoAngularRenderer::Render() {
  CHECK(is_initialized_) << "Initialize renderer first.";
  CHECK(renderer_->UseShader(shader_id_));

  SimpleTimer timer;

  renderer_->UpdatePose();
  Eigen::Matrix3f hmd_rotation =
      renderer_->GetHMDPose().block(0, 0, 3, 3).transpose();

  int best_frame = -1;
  double best_score = -1;
  Eigen::Vector3f lookat = -hmd_rotation.col(2);
  for (int i = 0; i < (1.0f / angular_resolution_) * 45; i++) {
    if (current_frame_ - i >= 0) {
      const double score = lookat.dot(frame_lookats_[current_frame_ - i]);
      if (score > best_score) {
        best_score = score;
        best_frame = current_frame_ - i;
      }
    }
    if (current_frame_ + i < frame_lookats_.size()) {
      const double score = lookat.dot(frame_lookats_[current_frame_ + i]);
      if (score > best_score) {
        best_score = score;
        best_frame = current_frame_ + i;
      }
    }
  }

  current_frame_ = best_frame;

  CHECK_GE(best_frame, 0);

  renderer_->UpdateTexture(frames_[best_frame], "image");

  Eigen::Matrix4f mvp = Eigen::Matrix4f::Identity();

  mvp.block(0, 0, 3, 3) *= hmd_rotation * frame_rotations_[best_frame];

  Eigen::Matrix4f mvp_left = renderer_->GetProjectionMatrix(0) * mvp;
  Eigen::Matrix4f mvp_right = renderer_->GetProjectionMatrix(1) * mvp;

  CHECK(renderer_->BindMesh(mesh_ids_[0]));
  renderer_->SetProjectionMatrix(mvp_left);
  renderer_->UploadShaderUniform(0, "right");
  renderer_->RenderEye(0);
  CHECK(renderer_->BindMesh(mesh_ids_[1]));
  renderer_->SetProjectionMatrix(mvp_right);
  renderer_->UploadShaderUniform(1, "right");
  renderer_->RenderEye(1);

  LOG(INFO) << "Rendered in " << timer.ElapsedTime() << "ms.";
}

}  // namespace replay
