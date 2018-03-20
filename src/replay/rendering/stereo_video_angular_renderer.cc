#include "replay/rendering/stereo_video_angular_renderer.h"
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/vr_context.h"
#include "replay/third_party/theia/sfm/types.h"
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
    "uniform sampler2DArray images;"
    "uniform int image_index;"
    "void main()\n"
    "{\n"
    "    color = texture(images, vec3(frag_uv, image_index)).rgb;"
    "}\n";
} // namespace

StereoVideoAngularRenderer::StereoVideoAngularRenderer(
    std::shared_ptr<VRContext> renderer)
    : renderer_(renderer), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize OpenGL renderer first!";
}

namespace {

void AngleAxisToLookAtUpvec(const Eigen::Vector3f &angle_axis,
                            Eigen::Vector3f &lookat, Eigen::Vector3f &up,
                            Eigen::Matrix3f &rotation) {
  Eigen::AngleAxisf aa(angle_axis.norm(), angle_axis.normalized());
  rotation = aa.toRotationMatrix();
  lookat = rotation.col(2);
  up = rotation.col(1);
}

} // namespace

bool StereoVideoAngularRenderer::Initialize(
    const std::string &spherical_video_filename) {
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(ERROR) << "Couldn't compile shader!";
    return false;
  }

  renderer_->SetViewportSize(1000, 1000);

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

  cv::Mat3b image;
  Eigen::Vector3f angle;

  static const int number_of_frames = 3;

  CHECK(renderer_->UseShader(shader_id_));
  renderer_->AllocateTextureArray("images", reader_.GetWidth(),
                                  reader_.GetHeight(), image.channels(),
                                  number_of_frames, number_of_frames > 100);
  frame_lookats_ =
      std::vector<Eigen::Vector3f>(number_of_frames, Eigen::Vector3f(0, 0, 0));
  frame_upvecs_ = std::vector<Eigen::Vector3f>(number_of_frames);
  frame_rotations_ = std::vector<Eigen::Matrix3f>(number_of_frames);
  LOG(INFO) << "Uploading frames...";

  int index = 0;
  while (reader_.GetOrientedFrame(image, angle)) {
    if (index == number_of_frames) {
      break;
    }

    AngleAxisToLookAtUpvec(angle, frame_lookats_[index], frame_upvecs_[index],
                           frame_rotations_[index]);
    bool skip_this_frame = false;
    for (int i = 0; i < index; i++) {
      if (frame_lookats_[i].dot(frame_lookats_[index]) > 0.7) {
        skip_this_frame = true;
        break;
      }
    }
    if (skip_this_frame) {
      LOG(INFO) << "Skipping frame!";
      frame_lookats_[index] = Eigen::Vector3f(0, 0, 0);
      continue;
    }
    renderer_->UploadTextureToArray(image, "images", index);
    index++;
  }

  LOG(INFO) << "Done";

  return true;
}

void StereoVideoAngularRenderer::Render() {
  CHECK(is_initialized_) << "Initialize renderer first.";
  CHECK(renderer_->UseShader(shader_id_));
  renderer_->ToggleCompanionWindow(true);
  renderer_->UpdatePose();
  Eigen::Matrix3f hmd_rotation =
      renderer_->GetHMDPose().block(0, 0, 3, 3).transpose();

  int best_frame = -1;
  double best_score = -1;
  const Eigen::Vector3f lookat = hmd_rotation.col(2);
  for (int i = 0; i < frame_lookats_.size(); i++) {
    const double score = lookat.dot(frame_lookats_[i]);
    if (score > best_score) {
      best_score = score;
      best_frame = i;
    }
  }
  CHECK_GE(best_frame, 0);

  Eigen::Matrix4f mvp = Eigen::Matrix4f::Identity();
  Eigen::Matrix3f inverse_frame_rotation =
      frame_rotations_[best_frame].inverse();
  mvp.block(0, 0, 3, 3) *= hmd_rotation * inverse_frame_rotation;

  Eigen::Matrix4f mvp_left = renderer_->GetProjectionMatrix(0) * mvp;
  LOG(INFO) << mvp_left;
  Eigen::Matrix4f mvp_right = renderer_->GetProjectionMatrix(1) * mvp;

  renderer_->UploadShaderUniform(best_frame, "image_index");
  renderer_->UploadMesh(meshes_[0]);
  renderer_->SetProjectionMatrix(mvp_left);
  renderer_->UploadShaderUniform(0, "right");
  renderer_->RenderEye(0);
  renderer_->UploadMesh(meshes_[1]);
  renderer_->SetProjectionMatrix(mvp_right);
  renderer_->UploadShaderUniform(1, "right");
  renderer_->RenderEye(1);
}

} // namespace replay
