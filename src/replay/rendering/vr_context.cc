#include "replay/rendering/vr_context.h"
#include <Eigen/Dense>
#include <chrono>
#include "glog/logging.h"
#include "openvr.h"

namespace replay {

namespace {
static const std::string companion_fragment =
    "#version 410\n"
    "out vec3 color;\n"
    "uniform sampler2D left;"
    "uniform sampler2D right;"
    "uniform vec2 window_size;"
    "void main()\n"
    "{\n"
    " vec2 screen_coords = vec2(gl_FragCoord.x / window_size.x, gl_FragCoord.y "
    "/ window_size.y);"
    " if (screen_coords.x < 0.5) {"
    " color = texture(left, vec2(screen_coords.x * 2, screen_coords.y)).rgb;"
    "}"
    " else {"
    " color = texture(right, vec2((screen_coords.x - 0.5) * 2 , "
    "screen_coords.y)).rgb;"
    "}"
    "}\n";
}  // namespace

VRContext::VRContext()
    : OpenGLContext(),
      keyboard_pitch_(0.0f),
      keyboard_yaw_(0.0f),
      keyboard_translation_(0, 0, 0),
      emulated_hmd_(false),
      hmd_viewport_width_(0),
      hmd_viewport_height_(0),
      hmd_index_(-1),
      companion_window_enabled_(false),
      companion_window_shader_(-1),
      companion_width_(2000),
      companion_height_(1000),
      vr_initialized_(false) {}

bool VRContext::InitializeHMD() {
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

  LOG(INFO) << "HMD initialized!";
  return true;
}

void VRContext::UpdatePoseFromKeyboard(int key, int action, int modifier) {
  if (action == GLFW_RELEASE) {
    return;
  }
  switch (key) {
    case GLFW_KEY_UP:
      keyboard_pitch_ += 0.005;
      break;
    case GLFW_KEY_DOWN:
      keyboard_pitch_ -= 0.005;
      break;
    case GLFW_KEY_RIGHT:
      keyboard_yaw_ -= 0.005;
      break;
    case GLFW_KEY_LEFT:
      keyboard_yaw_ += 0.005;
      break;
    case GLFW_KEY_W:
      keyboard_translation_[2] += 0.05;
      break;
    case GLFW_KEY_A:
      keyboard_translation_[0] -= 0.05;
      break;
    case GLFW_KEY_S:
      keyboard_translation_[2] -= 0.05;
      break;
    case GLFW_KEY_D:
      keyboard_translation_[0] -= 0.05;
      break;
  }
}

bool VRContext::Initialize() {
  if (!OpenGLContext::Initialize()) {
    return false;
  }
  ToggleCompanionWindow(false);
  if (!vr::VR_IsHmdPresent() || !InitializeHMD()) {
    LOG(INFO) << "Initializing HMD emulator...";
    ToggleCompanionWindow(true);
    SetKeyboardCallback(std::bind(&VRContext::UpdatePoseFromKeyboard, this,
                                  std::placeholders::_1, std::placeholders::_2,
                                  std::placeholders::_3));
    emulated_hmd_ = true;
  }

  // Compile companion window shader
  if (!CompileFullScreenShader(companion_fragment, &companion_window_shader_)) {
    return false;
  }

  vr_initialized_ = true;

  return true;
}

void VRContext::ToggleCompanionWindow(const bool enable) {
  DCHECK(is_initialized_) << "OpenGL context was not initialized.";
  if (enable) {
    ShowWindow();
  } else {
    HideWindow();
  }
  companion_window_enabled_ = enable;
}

Eigen::Matrix4f VRContext::GetProjectionMatrix(const int eye_id) const {
  DCHECK(vr_initialized_) << "Initialize VR context first.";
  if (emulated_hmd_) {
    Eigen::Matrix4f projection;
    // Using the default projection matrix from the HTC Vive
    projection << 0.755837, 0, -0.0563941, 0, 0, 0.680395, -0.00309659, 0, 0, 0,
        -1.00503, -0.0100503, 0, 0, -1, 0;
    return projection;
  }
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
  DCHECK(vr_initialized_) << "Initialize VR context first.";
  if (!emulated_hmd_) {
    vr::VRCompositor()->WaitGetPoses(device_poses_,
                                     vr::k_unMaxTrackedDeviceCount, NULL, 0);
    vr::HmdMatrix34_t pose =
        device_poses_[hmd_index_].mDeviceToAbsoluteTracking;
    hmd_pose_ << pose.m[0][0], pose.m[0][1], pose.m[0][2], pose.m[0][3],
        pose.m[1][0], pose.m[1][1], pose.m[1][2], pose.m[1][3], pose.m[2][0],
        pose.m[2][1], pose.m[2][2], pose.m[2][3], 0, 0, 0, 1;
  }
}

void VRContext::Render() { LOG(FATAL) << "Not implemented."; }

Eigen::Matrix4f VRContext::GetHMDPose() const {
  DCHECK(vr_initialized_) << "Initialize VR context first.";
  if (!emulated_hmd_) {
    return hmd_pose_;
  } else {
    Eigen::Matrix3f yaw;
    yaw << cos(keyboard_yaw_), 0, sin(keyboard_yaw_), 0, 1, 0,
        -sin(keyboard_yaw_), 0, cos(keyboard_yaw_);
    Eigen::Matrix3f pitch;
    pitch << 1, 0, 0, 0, cos(keyboard_pitch_), -sin(keyboard_pitch_), 0,
        sin(keyboard_pitch_), cos(keyboard_pitch_);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block(0, 0, 3, 3) = yaw * pitch;
    pose.block(0, 3, 3, 1) = keyboard_translation_;
    return pose;
  }
}

void VRContext::SetCompanionWindowSize(const int width, const int height) {
  DCHECK_GT(width, 0);
  DCHECK_GT(height, 0);

  companion_width_ = width;
  companion_height_ = height;
}

void VRContext::RenderEye(const int eye_id) {
  DCHECK(vr_initialized_) << "Initialize VR context first.";
  DCHECK_LE(eye_id, 1);
  DCHECK_GE(eye_id, 0);

  if (emulated_hmd_) {
    SetViewportSize(companion_width_, companion_height_);
  } else {
    SetViewportSize(hmd_viewport_width_, hmd_viewport_height_, false);
  }

  RenderToImage(&image_);

  const int shader = current_program_;

  const std::string eye_name = (eye_id == 0 ? "left" : "right");

  if (!emulated_hmd_) {
    UploadTexture(image_, eye_name);
    vr::Texture_t eye_texture = {(void *)(uintptr_t)GetTextureId(eye_name),
                                 vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
    vr::VRCompositor()->Submit(eye_id == 0 ? vr::Eye_Left : vr::Eye_Right,
                               &eye_texture);
    int error = glGetError();
    if (error != 0) {
      LOG(ERROR) << "VR Compositor caused OpenGL error " << error
                 << " upon Submit()";
    }
  }
  if (companion_window_enabled_) {
    SetViewportSize(companion_width_, companion_height_);
    UseShader(companion_window_shader_);
    UploadTexture(image_, eye_name);
    UploadShaderUniform(Eigen::Vector2f(companion_width_, companion_height_),
                        "window_size");
    OpenGLContext::Render();
    UseShader(shader);
  }
}

}  // namespace replay
