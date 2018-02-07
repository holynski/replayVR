#include "spherical_video_reader.h"
#include <glog/logging.h>
#include <Eigen/Sparse>
#include <iostream>
#include <opencv2/core.hpp>
#include <unordered_map>
#include "video_reader.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include "libavcodec/avcodec.h"
#include "libavutil/avutil.h"
}

namespace replay {

SphericalVideoReader::SphericalVideoReader(const std::string& filename) {
  CHECK(reader_.Open(filename)) << "Couldn't open video reader.";
  reader_.GetMesh("test");
}

bool SphericalVideoReader::Seek(const unsigned int time_ms) { return false; }

bool SphericalVideoReader::GetOrientedFrame(cv::Mat3b& frame,
                                            Eigen::Vector3f& angle_axis) {
  frame = reader_.ReadFrame();
  if (frame.empty()) {
    return false;
  }
  const uint16_t* buffer_data =
      reinterpret_cast<const uint16_t*>(reader_.ReadMetadataPacket());
  if (buffer_data == nullptr) {
    LOG(ERROR) << "No metadata packet available.";
    return false;
  }
  const CameraMotionMetadataAngleAxis* meta =
      reinterpret_cast<const CameraMotionMetadataAngleAxis*>(buffer_data);
  if (meta->reserved != 0) {
    LOG(ERROR) << "Metadata is invalid!";
    return false;
  }
  if (meta->type != 0) {
    LOG(ERROR) << "Only angle-axis supported!";
    return false;
  }
  angle_axis = Eigen::Vector3f(meta->angle_axis[0], meta->angle_axis[1],
                               meta->angle_axis[2]);
  return true;
}

int SphericalVideoReader::GetVideoLength() const { return 0; }
}
