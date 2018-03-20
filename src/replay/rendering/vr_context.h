#pragma once

#include <Eigen/Dense>
#include "openvr.h"
#include "replay/rendering/opengl_context.h"

namespace replay {

// A class to simplify VR HMD rendering.
//
// Sample usage:
//
// std::shared_ptr<VRContext> context = std::make_shared<VRContext>();
// context->Initialize();
// context->UseShader(s_id) // a shader you've compiled (see opengl_context.h)
// context->UploadMesh(mesh); while (true) {
//    context->Render();
// }
class VRContext : public OpenGLContext {
 public:
  //
  //
  // Functions for standard HMD rendering (rendering a mesh using a single
  // shader in VR):
  //
  //

  // An override of the default OpenGLContext Initialize() function which also
  // sets up the VR environment. 
  // If a VR HMD is not available, this function will initialize an HMD emulator as follows:
  //   - The companion window will be enabled automatically.
  //   - Manual controls will be enabled, allowing navigation with the keyboard 
  //     (WASD = translate, arrows/mouse = rotate)
  bool Initialize();

  // Enables or disables the companion window, which shows the HMD content on
  // the monitor.
  void ToggleCompanionWindow(const bool enable);
  void SetCompanionWindowSize(const int width, const int height);

  // Polls the HMD for the current pose, and then renders both eyes to the HMD.
  // Both eyes will render the same global geometry, and will use the same
  // shader and uniform values. Default to using this function for rendering unless you
  // need to:
  //     - Render a different mesh for each eye
  //     - Have different shader / shader uniforms for each eye
  //     - Have different OpenGL configurations for each eye
  //     - Use custom projection/view matrices.
  // If you need any of those things, use RenderEye() instead
  void Render();

  //
  //
  // Functions for advanced HMD rendering (rendering different
  // shaders/meshes/uniforms for each eye):
  //
  //

  // Fetches the most recent HMD and controller poses. 
  // If using the advanced functionality, this should be called before each pair of left/right RenderEye() calls.
  // If using an emulator, this function does nothing.
  void UpdatePose();

  // Renders a single eye to the HMD. This call is intended for more advanced
  // stereoscopic rendering, for instance when rendering different meshes for
  // each eye, different shaders, uniform values,  or not applying default
  // projection or pose matrices.
  //
  // This will not update the pose, nor will it upload a MVP matrix to the
  // shader. Matching left-right calls to this function should be preceded by a
  // call to UpdatePose(). The workflow should be:
  //
  // while (true) {
  //    UpdatePose();
  //    // set rendering options for left eye
  //    // upload left eye MVP
  //    RenderEye(0);
  //    // set rendering options for right eye
  //    // upload right eye MVP
  //    RenderEye(1);
  // }
  void RenderEye(const int eye_id);

  // Returns the 3x4 rotation&translation matrix of the global coordinate system
  // to the coordinate system of the camera.
  Eigen::Matrix4f GetHMDPose() const;

  // Returns the 4x4 projection matrix of a particular eye in OpenGL coordinates
  Eigen::Matrix4f GetProjectionMatrix(const int eye_id) const;

 private:
  bool InitializeHMD();
  void BindKeyboardControls();
  void UpdatePoseFromKeyboard(int key, int action, int modifier);
  vr::IVRSystem* hmd_;
  Eigen::Matrix4f left_projection_;
  Eigen::Matrix4f right_projection_;
  float keyboard_pitch_ = 0.0f;
  float keyboard_yaw_ = 0.0f;
  Eigen::Matrix3f keyboard_rotation_;
  Eigen::Vector3f keyboard_translation_;
  bool emulated_hmd_ = false;
  uint32_t hmd_viewport_width_;
  uint32_t hmd_viewport_height_;
  vr::TrackedDevicePose_t device_poses_[vr::k_unMaxTrackedDeviceCount];
  Eigen::Matrix4f hmd_pose_;
  int hmd_index_;
  bool companion_window_enabled_;
  int companion_window_shader_;
  int companion_width_ = 2000;
  int companion_height_ = 1000;
  cv::Mat3b image_;
};

}  // namespace replay
