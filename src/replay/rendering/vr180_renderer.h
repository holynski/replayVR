#pragma once

#include <chrono>
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/vr_context.h"
#include "replay/third_party/theia/sfm/camera/camera.h"
#include "replay/vr_180/vr_180_video_reader.h"

#ifdef __APPLE__
#define GLFW_INCLUDE_GLCOREARB
#else  // __APPLE__
#include <GL/glew.h>
#endif  // __APPLE__
#include <GLFW/glfw3.h>
namespace replay {

class VR180Renderer {
 public:
  explicit VR180Renderer(std::shared_ptr<VRContext> renderer);

  void Render();

  bool Initialize(const std::string &spherical_video_filename);

 private:
  std::shared_ptr<VRContext> renderer_;
  std::vector<Mesh> meshes_;
  int shader_id_;
  bool is_initialized_;
  cv::Mat3b image_;
  VR180VideoReader reader_;
  std::chrono::time_point<std::chrono::system_clock> last_frame_time;
};
}  // namespace replay
