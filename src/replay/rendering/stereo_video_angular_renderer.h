#ifndef REPLAY_MESH_STEREO_VIDEO_ANGULAR_RENDERER_H_
#define REPLAY_MESH_STEREO_VIDEO_ANGULAR_RENDERER_H_

#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/opengl_renderer.h"
#include "replay/third_party/theia/sfm/camera/camera.h"
#include "replay/vr_180/vr_180_video_reader.h"

#ifdef __APPLE__
#define GLFW_INCLUDE_GLCOREARB
#else  // __APPLE__
#include <GL/glew.h>
#endif  // __APPLE__
#include <GLFW/glfw3.h>
namespace replay {

class StereoVideoAngularRenderer {
 public:
  explicit StereoVideoAngularRenderer(std::shared_ptr<OpenGLRenderer> renderer);

  bool RenderEye(theia::Camera camera, const int eye_id,
                 const Eigen::Vector3f& lookat);

  bool Initialize(const std::string& spherical_video_filename);

 private:
  std::shared_ptr<OpenGLRenderer> renderer_;
  std::vector<Mesh> meshes_;
  int shader_id_;
  bool is_initialized_;
  std::vector<Eigen::Vector3f> frame_lookats_;
  std::vector<Eigen::Vector3f> frame_upvecs_;
  cv::Mat3b image_;
  VR180VideoReader reader_;
};
}  // namespace replay

#endif  // REPLAY_MESH_STEREO_VIDEO_ANGULAR_RENDERER_H_
