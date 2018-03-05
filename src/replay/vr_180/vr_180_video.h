#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "replay/vr_180/vr_180_video_reader.h"
#include "replay/mesh/mesh.h"
#include "replay/mesh/mesh.h"

namespace replay {

class VR180Video {
 public:
  VR180Video(const int cache_size = 50);

  // Loads a spherical video from disk
  // Returns true if successful, and false otherwise.
  // Will return false if:
  //    - file doesn't exist
  //    - file does not have projection metadata
  //    - file does not have angular metadata
  // This function will not check the validity of all frames or metadata
  // entries. Calls to GetXXX() functions may fail with exceptions if the file
  // is in some way invalid.
  bool Load(const std::string& filename);

  // Returns the number of frames in the video. If the full video has not been
  // read, this may result in a seek operation, which can be costly. 
  int NumFrames();

  // Lookup functions to get the frame id from either angle or timestamp
  int GetFrameId(const Eigen::Vector3f& angle_axis) const;
  int GetFrameId(const double time_in_ms) const;

  // In order to be robust to very long videos and little available memory,
  // these functions will seek through the video file unless the particular
  // frame was recently accessed. This function will first search a LRU cache
  // (whose size is defined in constructor), and if the data is not available,
  // will seek through the video. Video seeking can be quite costly.

  // Returns the frame for a valid frame id [0, NumFrames)
  cv::Mat3b GetFrame(const int frame_id);

  // Returns the angle (in angle-axis format) for a valid frame id [0, NumFrames)
  Eigen::Vector3f GetAngle(const int frame_id);

  // Returns the time (in milliseconds) for a valid frame id [0, NumFrames)
  double GetTimestamp(const int frame_id);

  // Returns the projection mesh for a given eye.
  Mesh GetProjectionMesh(const int eye_id) const;

 private:
  bool loaded_;
  std::unordered_map<int, cv::Mat3b> image_cache_;
  std::unordered_map<int, Eigen::Vector3f> angle_cache_;
  std::unordered_map<int, double> time_cache_;
  std::vector<Mesh> projection_meshes_;
  VR180VideoReader reader_;
};
}
