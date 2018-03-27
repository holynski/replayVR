#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include "libavcodec/avcodec.h"
#include "libavutil/avutil.h"
}

namespace replay {

enum StreamType { VIDEO, AUDIO, METADATA, NONE };

struct Packet {
  int stream_id = -1;
  StreamType type = StreamType::NONE;
  double time_in_seconds;
  double duration_in_seconds;
};

struct VideoPacket : Packet {
  AVFrame* frame;
};

struct AudioPacket : Packet {
  void* audio_data;
};

struct MetadataPacket : Packet {
  void* metadata;
};

class VideoReader {
 public:
  // Constructor - initialized registers avcodec and avformat if not done
  // already.
  VideoReader();

  // Opens a video file. Returns true if everything went smoothly and false
  // otherwise.
  bool Open(const std::string& filename);

  // Just returns the next video frame. It will be empty (cv::Mat::empty() ==
  // true) if the end of the video has been reached.
  //
  // If the 'bgr' flag is disabled, the returned image will be in RGB format
  // instead of BGR. This is useful when you want to avoid extra pixel type
  // conversion when uploading to OpenGL.
  cv::Mat3b ReadFrame(bool bgr = true);

  // Returns the next packet in the video file. This may be from either a Video,
  // Audio, or Metadata stream. If the end of the stream has been reached, or
  // some other error has occurred, nullptr will be returned.
  Packet* ReadPacket();

  // Seeks to a time in the video. Returns true if successful or false if
  // outside the bounds of the video or otherwise unsuccessful.
  bool SeekToTime(const double time_in_seconds);
  bool SeekToFrame(const int frame_number);
  bool SeekToMetadata(const double time_in_seconds);

  // Returns the length of the video in seconds.
  unsigned int GetVideoLength() const;

  // Return the dimensions of the video frames.
  int GetWidth() const;
  int GetHeight() const;

 protected:
  cv::Mat3b AVFrameToMat(AVFrame* frame, bool bgr) const;
  bool file_open_ = false;
  unsigned int video_length_;
  AVFormatContext* format_context_ = nullptr;
  AVCodecContext* video_decoder_context_ = nullptr;
  AVCodec* video_decoder_ = nullptr;
  AVStream* video_stream_ = nullptr;
  AVStream* audio_stream_ = nullptr;
  AVStream* metadata_stream_ = nullptr;
  unsigned int width_, height_;
  int video_stream_idx_ = -1;
  int audio_stream_idx_ = -1;
  int metadata_stream_idx_ = -1;
  AVPixelFormat pixel_format_;
  int video_pts_ = 0;

  static bool registered_avcodec_;
};

}  // namespace replay
