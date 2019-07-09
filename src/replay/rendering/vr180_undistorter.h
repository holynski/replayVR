#pragma once

#include <chrono>
#include "replay/depth_map/depth_map.h"
#include "replay/geometry/mesh.h"
#include "replay/rendering/vr_context.h"
#include "replay/third_party/theia/sfm/camera/camera.h"
#include "replay/vr_180/vr_180_video_reader.h"
#include "replay/camera/camera.h"

#ifdef __APPLE__
#define GLFW_INCLUDE_GLCOREARB
#else  // __APPLE__
#include <GL/glew.h>
#endif  // __APPLE__
#include <GLFW/glfw3.h>
namespace replay {

class VR180Undistorter {
 public:
  explicit VR180Undistorter(std::shared_ptr<OpenGLContext> renderer, const Camera& camera);

  bool UndistortFrame(cv::Mat3b* left, cv::Mat3b* right);

  bool Open(const std::string &spherical_video_filename);

 private:
  const Camera& camera_;
  std::shared_ptr<OpenGLContext> renderer_;
  std::vector<Mesh> meshes_;
  std::vector<int> mesh_ids_;
  int shader_id_;
  bool is_initialized_;
  cv::Mat3b image_;
  VR180VideoReader reader_;
};
}  // namespace replay
