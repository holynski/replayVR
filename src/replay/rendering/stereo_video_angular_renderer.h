#pragma once

#include "replay/depth_map/depth_map.h"
#include "replay/geometry/mesh.h"
#include "replay/rendering/vr_context.h"
#include "replay/third_party/theia/sfm/camera/camera.h"
#include "replay/vr_180/vr_180_video_reader.h"

namespace replay {

class StereoVideoAngularRenderer {
public:
  explicit StereoVideoAngularRenderer(std::shared_ptr<VRContext> renderer);

  void Render();

  bool Initialize(const std::string &spherical_video_filename);

private:
  std::shared_ptr<VRContext> renderer_;
  std::vector<Mesh> meshes_;
  std::vector<int> mesh_ids_;
  int shader_id_;
  bool is_initialized_;
  std::vector<Eigen::Vector3f> frame_lookats_;
  std::vector<Eigen::Vector3f> frame_upvecs_;
  std::vector<Eigen::Matrix3f> frame_rotations_;
  std::unordered_map<int, cv::Mat3b> frames_;
  cv::Mat3b image_;
  VR180VideoReader reader_;
  int current_frame_ = -1;
  float angular_resolution_ = 0.2; // in degrees
};
} // namespace replay
