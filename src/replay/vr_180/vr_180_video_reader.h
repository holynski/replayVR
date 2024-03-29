#pragma once

#include <replay/io/video_reader.h>
#include <replay/geometry/mesh.h>
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
  // If the bgr flag is changed to false, then the image will be converted to
  // RGB before returning. This may cause some overhead.
  // If the frame_time pointer is passed to a valid integer, it will be filled
  // with the timestamp of the frame. This can be used for easily identifying
  // the frame.
  bool GetOrientedFrame(cv::Mat3b& frame, Eigen::Matrix3f& rotation,
                        double* frame_time = nullptr, const bool bgr = true);

  // Returns the stereo distortion meshes for each eye
  std::vector<Mesh> GetMeshes();

  // Returns a mesh with the camera frusta for an entire video sequence.
  // Useful for visualizing the camera motion that has been reconstructed.
  // While no information is actually produced for the camera translation,
  // this function assumes the camera is rotating at a fixed radius, for
  // visualization purposes
  Mesh GetTrajectoryMesh();

  // Advanced usage, to avoid wasting time on video decode when we only want the
  // metadata.
  //
  // Fetches the next oriented frame pair, but does not decode the video frame.
  bool FetchOrientedFrame();
  // Decodes and returns the frame.
  cv::Mat3b GetFetchedFrame(const bool bgr = true) const;
  // Returns the orientation for the frame.
  Eigen::Matrix3f GetFetchedOrientation() const;

 private:
  bool ParseAllMetadata();
  Eigen::AngleAxisf GetAngleAxis(const double& time_in_seconds);

  // A mapping from time in seconds to the angular direction of the camera at
  // that time.
  std::map<double, Eigen::AngleAxisf> angular_metadata_;
  VideoPacket* encoded_frame_;
  Eigen::Matrix3f fetched_orientation_;
};

}  // namespace replay
