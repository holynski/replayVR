#include "replay/rendering/vr_context.h"
#include "openvr.h"
#include "glog/logging.h"
#include <Eigen/Dense>

namespace replay {

namespace {
static const std::string companion_fragment =
    "#version 410\n"
    "out vec3 color;\n"
    "uniform sampler2D left;"
    "uniform sampler2D right;"
    "void main()\n"
    "{\n"
    "    if (gl_FragCoord.x > 0.5) {"
	"       color = texture(left, vec2(gl_FragCoord.x * 2, gl_FragCoord.y)).rgb;"
    "}"
    "    else {"
	"       color = texture(right, vec2(gl_FragCoord.x * 2, gl_FragCoord.y)).rgb;"
    "}"
    "}\n";
}  // namespace

bool VRContext::Initialize() {
  if (!OpenGLContext::Initialize()) {
    return false;
  }
  is_initialized_ = false;
  companion_window_enabled_ = false;
  vr::EVRInitError error = vr::VRInitError_None;

  hmd_ = vr::VR_Init(&error, vr::VRApplication_Scene);
  if (error != vr::VRInitError_None) {
    LOG(ERROR) << "Unable to initialize VR context.";
    return false;
  }
  vr::IVRRenderModels *renderModels =
      (vr::IVRRenderModels *)vr::VR_GetGenericInterface(
          vr::IVRRenderModels_Version, &error);

  if (!renderModels) {
    vr::VR_Shutdown();
    LOG(ERROR) << "Unable to initialize VR context.";
    return false;
  }

  hmd_->GetRecommendedRenderTargetSize(&hmd_viewport_width_,
                                      &hmd_viewport_height_);

  // Retrieve and store the projection matrices for each eye, assuming that they
  // won't be changing during the scope of this VRContext object.
  vr::HmdMatrix44_t mat = hmd_->GetProjectionMatrix(vr::Eye_Left, 0.01, 2);
  left_projection_ << mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3],
      mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3], mat.m[2][0],
      mat.m[2][1], mat.m[2][2], mat.m[2][3], mat.m[3][0], mat.m[3][1],
      mat.m[3][2], mat.m[3][3];

  mat = hmd_->GetProjectionMatrix(vr::Eye_Right, 0.01, 2);
  right_projection_ << mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3],
      mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3], mat.m[2][0],
      mat.m[2][1], mat.m[2][2], mat.m[2][3], mat.m[3][0], mat.m[3][1],
      mat.m[3][2], mat.m[3][3];

  // Find and store the index of the HMD object among the tracked objects, so we
  // don't have to search for it each time we want to render.
  int hmd_index_ = -1;
  for (int device_id = 0; device_id < vr::k_unMaxTrackedDeviceCount;
       ++device_id) {
    if (hmd_->GetTrackedDeviceClass(device_id) == vr::TrackedDeviceClass_HMD) {
      hmd_index_ = device_id;
      break;
    }
  }
  if (hmd_index_ < 0) {
    return false;
  }
  is_initialized_ = true;

  // Compile companion window shader
//  if (!CompileFullScreenShader(companion_fragment, &companion_window_shader_)) {
//    return false;
//  }



  return true;
}

void VRContext::ToggleCompanionWindow(const bool enable) {
  DCHECK(is_initialized_) << "Initialize renderer first.";
  if (enable) {
    ShowWindow();
  }
  else {
    HideWindow();
  }
  companion_window_enabled_ = enable;
}

Eigen::Matrix4f VRContext::GetProjectionMatrix(const int eye_id) const {
  DCHECK(is_initialized_) << "Initialize renderer first.";
  switch (eye_id) {
    case 0:
      return left_projection_;
    case 1:
      return right_projection_;
    default:
      LOG(FATAL) << "Invalid eye_id: " << eye_id;
      return Eigen::Matrix4f();
  }
}

void VRContext::UpdatePose() {
  DCHECK(is_initialized_) << "Initialize renderer first.";
  vr::VRCompositor()->WaitGetPoses(device_poses_, vr::k_unMaxTrackedDeviceCount,
                                   NULL, 0);
  vr::HmdMatrix34_t pose = device_poses_[hmd_index_].mDeviceToAbsoluteTracking;
  hmd_pose_ << pose.m[0][0], pose.m[0][1], pose.m[0][2], pose.m[0][3],
      pose.m[1][0], pose.m[1][1], pose.m[1][2], pose.m[1][3], pose.m[2][0],
      pose.m[2][1], pose.m[2][2], pose.m[2][3], 0, 0, 0, 1;
}

void VRContext::Render() {
  LOG(FATAL) << "Not implemented.";
}

Eigen::Matrix4f VRContext::GetHMDPose() const {
  DCHECK(is_initialized_) << "Initialize renderer first.";
  return hmd_pose_;
}

void VRContext::SetCompanionWindowSize(const int width, const int height) {
  DCHECK_GT(width, 0);
  DCHECK_GT(height, 0);

  companion_width_ = width;
  companion_height_ = height;
}

void VRContext::RenderEye(const int eye_id) {
  DCHECK(is_initialized_) << "Initialize renderer first.";
  DCHECK_LE(eye_id, 1);
  DCHECK_GE(eye_id, 0);

  SetViewportSize(hmd_viewport_width_, hmd_viewport_height_);

  if (image_.empty()) {
    image_ = cv::Mat3b(1, 1);
  }

  RenderToImage(&image_);

  
//  const int shader = current_program_;
 // UseShader(companion_window_shader_);
  SetViewportSize(companion_width_, companion_height_);
  const std::string eye_name = (eye_id == 0 ? "left" : "right");
  UploadTexture(image_, eye_name);
  vr::Texture_t eye_texture = {(void *)(uintptr_t)GetTextureId(eye_name),
                               vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
  vr::VRCompositor()->Submit(eye_id == 0 ? vr::Eye_Left : vr::Eye_Right,
                             &eye_texture);
  int error = glGetError();
  if (error != 0) {
	  LOG(ERROR) << "VR Compositor caused OpenGL error " << error << " upon Submit()";
  }
  if (companion_window_enabled_) {
    OpenGLContext::Render();
  }

  // Return to the shader that was being used before the function was called
//  UseShader(shader);
}
}  // namespace replay
