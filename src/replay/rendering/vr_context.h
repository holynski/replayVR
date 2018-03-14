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
  bool Initialize();

  // Enables or disabled the companion window, which shows the HMD content on
  // the monitor.
  void ToggleCompanionWindow(const bool enable);
  void SetCompanionWindowSize(const int width, const int height);

  // Polls the HMD for the current pose, and then renders both eyes to the HMD.
  // Both eyes will render the same global geometry, and will use the same
  // shader and uniform values. Use this function for rendering unless you
  // otherwise need to access the OpenGL context between rendering each eye (for
  // changing the mesh, uniforms, or shader).
  void Render();

  //
  //
  // Functions for advanced HMD rendering (rendering different
  // shaders/meshes/uniforms for each eye):
  //
  //

  // Fetches the most recent HMD and controller poses.
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

  Eigen::Matrix4f GetProjectionMatrix(const int eye_id) const;

 private:
  vr::IVRSystem* hmd_;
  Eigen::Matrix4f left_projection_;
  Eigen::Matrix4f right_projection_;
  uint32_t hmd_viewport_width_;
  uint32_t hmd_viewport_height_;
  vr::TrackedDevicePose_t device_poses_[vr::k_unMaxTrackedDeviceCount];
  Eigen::Matrix4f hmd_pose_;
  int hmd_index_;
  bool companion_window_enabled_;
  int companion_window_shader_;
  int companion_width_ = 1000;
  int companion_height_ = 500;
  cv::Mat3b image_;
};

}  // namespace replay
