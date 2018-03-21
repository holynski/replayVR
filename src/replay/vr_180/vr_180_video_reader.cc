#include "vr_180_video_reader.h"
#include <glog/logging.h>
#include <replay/io/video_reader.h>
#include <replay/mesh/mesh.h>
#include <replay/util/matrix_utils.h>
#include <replay/vr_180/mesh_projection_parser.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/core.hpp>
#include <unordered_map>

namespace replay {

bool VR180VideoReader::Open(const std::string &filename) {
  VideoReader::Open(filename);

  if (!ParseAllMetadata()) {
    LOG(ERROR) << "Metadata not found in stream.";
    return false;
  }
  return true;
}

bool VR180VideoReader::GetOrientedFrame(cv::Mat3b &frame,
                                        Eigen::Matrix3f &rotation) {
  CHECK(file_open_) << "Call Open() first!";

  Packet *packet;
  while ((packet = ReadPacket()) && packet->stream_id != video_stream_idx_) {
    if (packet->stream_id == -1) {
      return false;
    }
  }

  if (!packet) {
    return false;
  }

  VideoPacket *video_packet = static_cast<VideoPacket *>(packet);
  frame = AVFrameToMat(video_packet->frame);
  const Eigen::AngleAxisf &angle_axis =
      GetAngleAxis(video_packet->time_in_seconds);

  rotation = angle_axis.toRotationMatrix().transpose();
  Eigen::Vector3f yaw_pitch_roll = DecomposeRotation(rotation);
  yaw_pitch_roll[0] *= -1;
  rotation = ComposeRotation(yaw_pitch_roll);

  return true;
}

std::vector<Mesh> VR180VideoReader::GetMeshes() {
  CHECK(file_open_) << "Call Open() first!";
  uint8_t *side_data = nullptr;
  CHECK_NOTNULL((side_data = av_stream_get_side_data(
                     video_stream_, AV_PKT_DATA_SPHERICAL, nullptr)));
  AVSphericalMapping *mapping =
      reinterpret_cast<AVSphericalMapping *>(side_data);

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
      const uint16_t *buffer_data =
          reinterpret_cast<const uint16_t *>(packet.data);
      if (buffer_data == nullptr) {
        LOG(ERROR) << "No metadata packet available.";
        return false;
      }
      const CameraMotionMetadataAngleAxis *meta =
          reinterpret_cast<const CameraMotionMetadataAngleAxis *>(buffer_data);
      if (meta->reserved != 0) {
        LOG(ERROR) << "Metadata is invalid!";
        return false;
      }
      if (meta->type != 0) {
        LOG(ERROR) << "Only angle-axis supported!";
        return false;
      }
      Eigen::Vector3f vector(meta->angle_axis[0], meta->angle_axis[1],
                             meta->angle_axis[2]);

      angular_metadata_[packet.pts *
                        static_cast<double>(metadata_stream_->time_base.num) /
                        metadata_stream_->time_base.den] =
          Eigen::AngleAxisf(vector.norm(), vector.normalized());
    }
  }
  SeekToFrame(0);
  LOG(INFO) << "Found " << angular_metadata_.size() << " metadata entries!";
  return !angular_metadata_.empty();
}

Eigen::AngleAxisf VR180VideoReader::GetAngleAxis(const double &time) {
  double min_distance = 1;
  Eigen::AngleAxisf min_entry;
  double best_time = 0;
  for (auto &entry : angular_metadata_) {
    const double distance = std::abs(entry.first - time);
    if (distance < min_distance) {
      min_distance = distance;
      min_entry = entry.second;
      best_time = entry.first;
    }
  }
  return min_entry;
}

Mesh VR180VideoReader::GetTrajectoryMesh() {
  CHECK(file_open_) << "Call Open() first!";
  SeekToMetadata(0);
  Mesh mesh;
  static const float head_rotation_radius = 10.0f;
  static const float pyramid_height = 0.5f;
  static const float pyramid_width = 0.1f;
  Eigen::Matrix3f rotation;
  Eigen::Vector3f lookat, up, left;
  cv::Mat3b image;
  while (GetOrientedFrame(image, rotation)) {
    lookat = -rotation.col(2);
    up = rotation.col(1);
    left = rotation.col(0);

    std::vector<VertexId> ids(5);
    ids[0] = mesh.AddVertex(lookat * head_rotation_radius);
    ids[1] =
        mesh.AddVertex((lookat * head_rotation_radius) +
                       (pyramid_height * lookat) + pyramid_width * (up + left));
    ids[2] =
        mesh.AddVertex((lookat * head_rotation_radius) +
                       (pyramid_height * lookat) + pyramid_width * (up - left));
    ids[3] = mesh.AddVertex((lookat * head_rotation_radius) +
                            (pyramid_height * lookat) +
                            pyramid_width * (-up + left));
    ids[4] = mesh.AddVertex((lookat * head_rotation_radius) +
                            (pyramid_height * lookat) +
                            pyramid_width * (-up - left));
    mesh.AddTriangleFace(ids[0], ids[2], ids[1]);
    mesh.AddTriangleFace(ids[0], ids[4], ids[2]);
    mesh.AddTriangleFace(ids[0], ids[1], ids[3]);
    mesh.AddTriangleFace(ids[0], ids[3], ids[4]);
    mesh.AddTriangleFace(ids[1], ids[2], ids[3]);
    mesh.AddTriangleFace(ids[3], ids[2], ids[4]);
  }
  return mesh;
}

}  // namespace replay
