#include "replay/vr_180/vr_180_video.h"

#include <glog/logging.h>

namespace replay {

VR180Video::VR180Video(const int cache_size) : loaded_(false) {}

bool VR180Video::Load(const std::string& filename) {
  if (!reader_.Open(filename)) {
    LOG(ERROR) << "Couldn't open spherical video " << filename;
    return false;
  }
  projection_meshes_ = reader_.GetMeshes();
  if (projection_meshes_.size() != 2 || projection_meshes_[0].NumVertices() == 0 || projection_meshes_[0].NumTriangleFaces() == 0 || projection_meshes_[1].NumVertices() == 0 || projection_meshes_[1].NumTriangleFaces() == 0) {

    LOG(ERROR) << "Projection meshes were invalid.";
    return false;
  }

  loaded_ = true;
  return true;
}

int VR180Video::NumFrames() { 
  CHECK(loaded_) << "Call Load() first!";
  LOG(FATAL) << "Function not implemented."; 
}

int VR180Video::GetFrameId(const Eigen::Vector3f& angle_axis) const {
  CHECK(loaded_) << "Call Load() first!";
  LOG(FATAL) << "Function not implemented.";
}

int VR180Video::GetFrameId(const double time_in_ms) const {
  CHECK(loaded_) << "Call Load() first!";

  LOG(FATAL) << "Function not implemented.";
}

cv::Mat3b VR180Video::GetFrame(const int frame_id) {
  CHECK(loaded_) << "Call Load() first!";
  LOG(FATAL) << "Function not implemented.";
}

Eigen::Vector3f VR180Video::GetAngle(const int frame_id) {
  CHECK(loaded_) << "Call Load() first!";
  LOG(FATAL) << "Function not implemented.";
}

double VR180Video::GetTimestamp(const int frame_id) {
  CHECK(loaded_) << "Call Load() first!";
  LOG(FATAL) << "Function not implemented.";
}

}
