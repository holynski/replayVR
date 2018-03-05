#pragma once

#include <Eigen/Sparse>
#include <iostream>
#include <opencv2/core.hpp>
#include <map>
#include <replay/io/video_reader.h>
#include <replay/mesh/mesh.h>

namespace replay {

class CameraMotionMetadataAngleAxis {
 public:
  uint16_t reserved;
  uint16_t type;
  float angle_axis[3];
};

class VR180VideoReader : public VideoReader {
 public:
  bool Open(const std::string& filename);
  bool GetOrientedFrame(cv::Mat3b& frame, Eigen::Vector3f& angle_axis);
  std::vector<Mesh> GetMeshes();
 private:
  bool ParseAllMetadata();
  Eigen::Vector3f GetAngleAxis(const double& time_in_seconds);

  // A mapping from time in seconds to the angular direction of the camera at that time.
  std::map<double, Eigen::Vector3f> angular_metadata_;
};

}  // namespace replay
