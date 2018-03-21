#pragma once

#include <replay/io/video_reader.h>
#include <replay/mesh/mesh.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>

namespace replay {

class CameraMotionMetadataAngleAxis {
 public:
  uint16_t reserved;
  uint16_t type;
  float angle_axis[3];
};

class VR180VideoReader : public VideoReader {
 public:
  // Opens a VR180 video. Returns false if the file does not conform to the
  // VR180 metadata format.
  bool Open(const std::string& filename);

  // Returns a frame with its corresponding rotation (world -> local)
  bool GetOrientedFrame(cv::Mat3b& frame, Eigen::Matrix3f& rotation);

  // Returns the stereo distortion meshes for each eye
  std::vector<Mesh> GetMeshes();

  // Returns a mesh with the camera frusta for an entire video sequence.
  // Useful for visualizing the camera motion that has been reconstructed.
  // While no information is actually produced for the camera translation,
  // this function assumes the camera is rotating at a fixed radius, for
  // visualization purposes
  Mesh GetTrajectoryMesh();

 private:
  bool ParseAllMetadata();
  Eigen::AngleAxisf GetAngleAxis(const double& time_in_seconds);

  // A mapping from time in seconds to the angular direction of the camera at
  // that time.
  std::map<double, Eigen::AngleAxisf> angular_metadata_;
};

}  // namespace replay
