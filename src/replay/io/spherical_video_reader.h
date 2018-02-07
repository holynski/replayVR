#pragma once

#include <Eigen/Sparse>
#include <iostream>
#include <opencv2/core.hpp>
#include <unordered_map>
#include "video_reader.h"

namespace replay {

class CameraMotionMetadataAngleAxis {
 public:
  uint16_t reserved;
  uint16_t type;
  float angle_axis[3];
};

class SphericalVideoReader {
 public:
  SphericalVideoReader(const std::string& filename);
  bool GetOrientedFrame(cv::Mat3b& frame, Eigen::Vector3f& angle_axis);
  bool Seek(const unsigned int time_ms);
  int GetVideoLength() const;

  // TODO(holynski): Function for getting mesh

 private:
  VideoReader reader_;
};

}  // namespace replay
