#include "vr_180_video_reader.h"
#include <glog/logging.h>
#include <replay/io/video_reader.h>
#include <replay/mesh/mesh.h>
#include <replay/vr_180/mesh_projection_parser.h>
#include <Eigen/Sparse>
#include <iostream>
#include <opencv2/core.hpp>
#include <unordered_map>

namespace replay {

bool VR180VideoReader::Open(const std::string& filename) {
  VideoReader::Open(filename);

  if (!ParseAllMetadata()) {
    LOG(ERROR) << "Metadata not found in stream.";
    return false;
  }
  return true;
}

bool VR180VideoReader::GetOrientedFrame(cv::Mat3b& frame,
                                        Eigen::Vector3f& angle_axis) {
  CHECK(file_open_) << "Call Open() first!";

  Packet* packet;
  while ((packet = ReadPacket()) && packet->stream_id != video_stream_idx_) {
    if (packet->stream_id == -1) {
      return false;
    }
  }

  if (!packet) {
    return false;
  }

  VideoPacket* video_packet = static_cast<VideoPacket*>(packet);
  frame = AVFrameToMat(video_packet->frame);
  angle_axis = GetAngleAxis(video_packet->time_in_seconds);

  return true;
}

std::vector<Mesh> VR180VideoReader::GetMeshes() {
  CHECK(file_open_) << "Call Open() first!";
  uint8_t* side_data = nullptr;
  CHECK_NOTNULL((side_data = av_stream_get_side_data(
                     video_stream_, AV_PKT_DATA_SPHERICAL, nullptr)));
  AVSphericalMapping* mapping =
      reinterpret_cast<AVSphericalMapping*>(side_data);

  MeshProjectionParser parser;
  std::vector<Mesh> meshes =
      parser.Parse(mapping->mesh, mapping->mesh.encoding);
  return meshes;
}

bool VR180VideoReader::ParseAllMetadata() {
  SeekToMetadata(0);
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = NULL;
  packet.size = 0;
  packet.stream_index = -1;
  while (true) {
    if (av_read_frame(format_context_, &packet) < 0) {
      break;
    }
    if (packet.stream_index == metadata_stream_idx_) {
      const uint16_t* buffer_data =
          reinterpret_cast<const uint16_t*>(packet.data);
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
      angular_metadata_[packet.pts *
                        static_cast<double>(metadata_stream_->time_base.num) /
                        metadata_stream_->time_base.den] =
          Eigen::Vector3f(meta->angle_axis[0], meta->angle_axis[1],
                          meta->angle_axis[2]);
    }
  }
  SeekToFrame(0);
  LOG(INFO) << "Found " << angular_metadata_.size() << " metadata entries!";
  return !angular_metadata_.empty();
}

Eigen::Vector3f VR180VideoReader::GetAngleAxis(const double& time) {
  double min_distance = 1;
  Eigen::Vector3f min_entry(0, 0, 0);
  double best_time = 0;
  for (auto& entry : angular_metadata_) {
    const double distance = std::abs(entry.first - time);
    if (distance < min_distance) {
      min_distance = distance;
      min_entry = entry.second;
      best_time = entry.first;
    }
  }
  return min_entry;
}

}  // namespace replay
