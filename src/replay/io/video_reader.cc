extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include "libavcodec/avcodec.h"
#include "libavutil/avutil.h"
#include "libavutil/dict.h"
#include "libavutil/spherical.h"
#include "libswscale/swscale.h"
#include "zlib.h"
}

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "replay/io/video_reader.h"
#include "replay/mesh.h"

namespace replay {

// Static flag to ensure we only register libav once at startup
bool VideoReader::registered_avcodec_ = false;

VideoReader::VideoReader() {
  if (!registered_avcodec_) {
    avcodec_register_all();
    av_register_all();
    registered_avcodec_ = true;
  }
}

bool VideoReader::Open(const std::string& filename) {
  if (avformat_open_input(&format_context_, filename.c_str(), NULL, NULL) < 0) {
    LOG(ERROR) << "Couldn't open file and allocate context for " << filename;
    return false;
  }
  if (avformat_find_stream_info(format_context_, NULL) < 0) {
    LOG(ERROR) << "Couldn't find stream info.";
    return false;
  }
  AVDictionary* opts = NULL;

  if ((video_stream_idx_ = av_find_best_stream(
           format_context_, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0)) < 0) {
    LOG(ERROR) << "Couldn't find video stream.";
    return false;
  }
  metadata_stream_idx_ =
      av_find_best_stream(format_context_, AVMEDIA_TYPE_DATA, -1, -1, NULL, 0);

  video_stream_ = format_context_->streams[video_stream_idx_];

  if ((video_decoder_ = avcodec_find_decoder(
           video_stream_->codecpar->codec_id)) == nullptr) {
    LOG(ERROR) << "Couldn't find decoder.";
    return false;
  }

  if ((video_decoder_context_ = avcodec_alloc_context3(video_decoder_)) ==
      nullptr) {
    LOG(ERROR) << "Couldn't allocate a codec context.";
    return false;
  }

  if ((avcodec_parameters_to_context(video_decoder_context_,
                                     video_stream_->codecpar)) < 0) {
    LOG(ERROR) << "Couldn't set context parameters.";
    return false;
  }

  av_dict_set(&opts, "refcounted_frames", "0", 0);
  if (avcodec_open2(video_decoder_context_, video_decoder_, &opts) < 0) {
    LOG(ERROR) << "Unable to open video decoder.";
    return false;
  }

  width_ = video_decoder_context_->width;
  height_ = video_decoder_context_->height;
  pixel_format_ = video_decoder_context_->pix_fmt;
  CHECK_EQ(pixel_format_, AV_PIX_FMT_YUVJ420P);

  if (height_ <= 0 || width_ <= 0) {
    LOG(ERROR) << "Invalid video resolution: " << width_ << " " << height_;
    return false;
  }

  file_open_ = true;
  return true;
}

bool VideoReader::ReadUntilPacketFromStream(const int stream_index) {
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = NULL;
  packet.size = 0;
  packet.stream_index = -1;
  while (packet.stream_index != stream_index) {
    if (av_read_frame(format_context_, &packet) < 0) {
      return false;
    }
    if (packet.stream_index == metadata_stream_idx_) {
      metadata_.emplace_back(packet.pts,
                             reinterpret_cast<void*>(new uint8_t[packet.size]));
      memcpy(metadata_[metadata_.size() - 1].second, packet.data, packet.size);
    }
    if (packet.stream_index == video_stream_idx_) {
      if (avcodec_send_packet(video_decoder_context_, &packet) < 0) {
        return false;
      }

      AVFrame* frame = av_frame_alloc();
      if (avcodec_receive_frame(video_decoder_context_, frame) < 0) {
        return false;
      }

      cv::Mat rgb(height_, width_, CV_8UC3);
      AVFrame dst;
      dst.data[0] = (uint8_t*)rgb.data;
      avpicture_fill((AVPicture*)&dst, dst.data[0], AV_PIX_FMT_BGR24, width_,
                     height_);

      SwsContext* convert_ctx;
      AVPixelFormat src_pixfmt = static_cast<AVPixelFormat>(frame->format);
      AVPixelFormat dst_pixfmt = AV_PIX_FMT_BGR24;
      convert_ctx =
          sws_getContext(width_, height_, src_pixfmt, width_, height_,
                         dst_pixfmt, SWS_FAST_BILINEAR, NULL, NULL, NULL);

      CHECK_NOTNULL(convert_ctx);

      sws_scale(convert_ctx, frame->data, frame->linesize, 0, height_, dst.data,
                dst.linesize);

      frames_.emplace_back(packet.pts, rgb);
    }
  }
  return true;
}

cv::Mat3b VideoReader::ReadFrame() {
  CHECK(file_open_) << "Call Open() first!";

  // If we haven't buffered any frames, read ahead in the stream until we find
  // one.
  if (frames_.empty()) {
    if (!ReadUntilPacketFromStream(video_stream_idx_)) {
      return cv::Mat3b();
    }
  }

  // Return the first buffered frame and delete it from the cache.
  video_pts_ = frames_[0].first;
  cv::Mat3b return_frame = frames_[0].second;
  frames_.erase(frames_.begin());
  LOG(INFO) << "VideoPTS: " << video_pts_;
  return return_frame;
}

void* VideoReader::ReadMetadataPacket() {
  CHECK(file_open_) << "Call Open() first!";

  // If the metadata packet corresponding to the last returned video frame
  // hasn't been read already, advance the stream until it is found.
  // In this case, also delete all the metadata entries we have cached, since we
  // won't be needing those anymore.
  while (metadata_.empty() ||
         video_pts_ > metadata_[metadata_.size() - 1].first) {
    metadata_.erase(metadata_.begin(), metadata_.end());
    if (!ReadUntilPacketFromStream(metadata_stream_idx_)) {
      return nullptr;
    }
  }

  // Search the buffered metadata values until we find the one with the smallest
  // PTS difference to the video frame. We do this by iterating through the
  // buffered values and stopping once we see that the absolute value in PTS
  // difference begins to increase. If this doesn't happen, we return the last
  // frame.
  std::pair<int, void*>* last_meta = &(metadata_[0]);
  int i = 0;
  for (auto& meta : metadata_) {
    if (std::abs(meta.first - video_pts_) >
        std::abs(last_meta->first - video_pts_)) {
      LOG(INFO) << "MetaPTS: " << last_meta->first;
      void* retval = last_meta->second;
      metadata_.erase(metadata_.begin(), metadata_.begin() + i);
      return retval;
    }
    i++;
    last_meta = &meta;
  }
  metadata_.erase(metadata_.begin(), metadata_.begin() + metadata_.size() - 1);
  LOG(INFO) << "MetaPTS: " << metadata_[metadata_.size() - 1].first;
  return metadata_[metadata_.size() - 1].second;
}

Mesh* VideoReader::GetMesh(const std::string& key) {
  CHECK(file_open_) << "Call Open() first!";
  uint8_t* side_data = nullptr;
  CHECK_NOTNULL((side_data = av_stream_get_side_data(
                     video_stream_, AV_PKT_DATA_SPHERICAL, nullptr)));
  AVSphericalMapping* mapping =
      reinterpret_cast<AVSphericalMapping*>(side_data);

  Mesh* mesh = new Mesh;
  char* encoding_chars = reinterpret_cast<char*>(&(mapping->mesh.encoding));
  LOG(INFO) << encoding_chars[0] << encoding_chars[1] << encoding_chars[2]
            << encoding_chars[3];
  CHECK(mesh->LoadFromSphericalMetadata(mapping->mesh)) << "Mesh was not valid";
  mesh->Save("/Users/holynski/testmesh.ply");
  return mesh;
}

bool VideoReader::Seek(const unsigned int time_in_ms) {
  CHECK(file_open_) << "Call Open() first!";
  LOG(FATAL) << "Function not implemented.";
  if (time_in_ms >= GetVideoLength()) {
    return false;
  }
  current_time_ms_ = time_in_ms;
  return true;
}

unsigned int VideoReader::GetVideoLength() const {
  CHECK(file_open_) << "Call Open() first!";
  LOG(FATAL) << "Function not implemented.";
  return video_length_;
}

int VideoReader::GetWidth() const {
  CHECK(file_open_) << "Call Open() first!";
  return width_;
}

int VideoReader::GetHeight() const {
  CHECK(file_open_) << "Call Open() first!";
  return height_;
}
}
