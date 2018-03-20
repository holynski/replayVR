extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/dict.h"
#include "libavutil/imgutils.h"
#include "libavutil/samplefmt.h"
#include "libavutil/spherical.h"
#include "libavutil/timestamp.h"
#include "libswscale/swscale.h"
#include "zlib.h"
}

#include "replay/io/video_reader.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

// Get video seeking working
//
// Get nearest metadata frame to video frame

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

bool VideoReader::Open(const std::string &filename) {
  if (avformat_open_input(&format_context_, filename.c_str(), NULL, NULL) < 0) {
    LOG(ERROR) << "Couldn't open file and allocate context for " << filename;
    return false;
  }
  if (avformat_find_stream_info(format_context_, NULL) < 0) {
    LOG(ERROR) << "Couldn't find stream info.";
    return false;
  }
  AVDictionary *opts = NULL;

  if ((video_stream_idx_ = av_find_best_stream(
           format_context_, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0)) < 0) {
    LOG(ERROR) << "Couldn't find video stream.";
    return false;
  }
  metadata_stream_idx_ =
      av_find_best_stream(format_context_, AVMEDIA_TYPE_DATA, -1, -1, NULL, 0);

  audio_stream_idx_ =
      av_find_best_stream(format_context_, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);

  video_stream_ = format_context_->streams[video_stream_idx_];
  audio_stream_ = format_context_->streams[audio_stream_idx_];
  metadata_stream_ = format_context_->streams[metadata_stream_idx_];

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

Packet *VideoReader::ReadPacket() {
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = NULL;
  packet.size = 0;
  packet.stream_index = -1;
  if (av_read_frame(format_context_, &packet) < 0) {
    return nullptr;
  }
  if (packet.stream_index == video_stream_idx_) {
    if (avcodec_send_packet(video_decoder_context_, &packet) < 0) {
      return nullptr;
    }

    AVFrame *frame = av_frame_alloc();
    if (avcodec_receive_frame(video_decoder_context_, frame) < 0) {
      return nullptr;
    }

    VideoPacket *return_packet = new VideoPacket;
    return_packet->stream_id = packet.stream_index;
    return_packet->type = StreamType::VIDEO;
    return_packet->frame = frame;
    return_packet->time_in_seconds =
        (packet.pts * static_cast<double>(video_stream_->time_base.num)) /
        video_stream_->time_base.den;
    return_packet->duration_in_seconds =
        (packet.duration * static_cast<double>(video_stream_->time_base.num)) /
        video_stream_->time_base.den;
    return return_packet;
  } else if (packet.stream_index == metadata_stream_idx_) {
    MetadataPacket *return_packet = new MetadataPacket;
    return_packet->stream_id = packet.stream_index;
    return_packet->type = StreamType::METADATA;
    return_packet->time_in_seconds =
        (packet.pts * static_cast<double>(metadata_stream_->time_base.num)) /
        metadata_stream_->time_base.den;
    return_packet->metadata =
        reinterpret_cast<void *>(new uint8_t[packet.size]);
    memcpy(return_packet->metadata, packet.data, packet.size);
    // return_packet->metadata = packet.data;
    return return_packet;
  } else if (packet.stream_index == audio_stream_idx_) {
    AudioPacket *return_packet = new AudioPacket;
    return_packet->stream_id = packet.stream_index;
    return_packet->type = StreamType::AUDIO;
    return_packet->time_in_seconds =
        (packet.pts * static_cast<double>(audio_stream_->time_base.num)) /
        audio_stream_->time_base.den;
    return return_packet;
  } else {
    Packet *return_packet = new Packet;
    return_packet->stream_id = packet.stream_index;
    return return_packet;
  }
} // namespace replay

cv::Mat3b VideoReader::ReadFrame() {
  CHECK(file_open_) << "Call Open() first!";

  Packet *packet;
  while ((packet = ReadPacket())->stream_id != video_stream_idx_) {
    if (packet->stream_id == -1) {
      // We hit the end of the stream.
      return cv::Mat3b();
    }
  }

  VideoPacket *video_packet = static_cast<VideoPacket *>(packet);
  return AVFrameToMat(video_packet->frame);
}

bool VideoReader::SeekToTime(const double time_in_seconds) {
  CHECK(file_open_) << "Call Open() first!";

  return SeekToFrame((time_in_seconds * video_stream_->r_frame_rate.num) /
                     video_stream_->r_frame_rate.den);
}

bool VideoReader::SeekToMetadata(const double time_in_seconds) {
  CHECK(file_open_) << "Call Open() first!";

  if (av_seek_frame(format_context_, metadata_stream_idx_,
                    (time_in_seconds * metadata_stream_->time_base.den) /
                        metadata_stream_->time_base.num,
                    AVSEEK_FLAG_ANY) < 0) {
    return false;
  }

  return true;
}

bool VideoReader::SeekToFrame(const int frame_number) {
  CHECK(file_open_) << "Call Open() first!";

  double time_in_seconds =
      frame_number * static_cast<double>(video_stream_->r_frame_rate.den) /
      video_stream_->r_frame_rate.num;

  if (av_seek_frame(format_context_, video_stream_idx_,
                    (time_in_seconds * video_stream_->time_base.den) /
                        video_stream_->time_base.num,
                    AVSEEK_FLAG_BACKWARD) < 0) {
    return false;
  }
  Packet *packet;
  while ((packet = ReadPacket())->stream_id != video_stream_idx_) {
    if (packet->stream_id == -1) {
      // We hit the end of the stream.
      return false;
    }
  }
  while (time_in_seconds >
         (packet->time_in_seconds + packet->duration_in_seconds)) {
    while ((packet = ReadPacket())->stream_id != video_stream_idx_) {
      if (packet->stream_id == -1) {
        // We hit the end of the stream.
        return false;
      }
    }
  }
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

cv::Mat3b VideoReader::AVFrameToMat(AVFrame *frame) const {
  cv::Mat3b retval = cv::Mat(height_, width_, CV_8UC3);
  AVFrame dst;
  dst.data[0] = (uint8_t *)retval.data;
  avpicture_fill((AVPicture *)&dst, dst.data[0], AV_PIX_FMT_BGR24, width_,
                 height_);

  SwsContext *convert_ctx;
  AVPixelFormat src_pixfmt = static_cast<AVPixelFormat>(frame->format);
  AVPixelFormat dst_pixfmt = AV_PIX_FMT_BGR24;
  convert_ctx = sws_getContext(width_, height_, src_pixfmt, width_, height_,
                               dst_pixfmt, SWS_FAST_BILINEAR, NULL, NULL, NULL);

  CHECK_NOTNULL(convert_ctx);

  sws_scale(convert_ctx, frame->data, frame->linesize, 0, height_, dst.data,
            dst.linesize);
  return retval;
}

} // namespace replay
