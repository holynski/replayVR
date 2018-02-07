#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#include "replay/mesh.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include "libavcodec/avcodec.h"
#include "libavutil/avutil.h"
}

namespace replay {

class VideoReader {
 public:
  // Constructor - initialized registers avcodec and avformat if not done
  // already.
  VideoReader();

  // Opens a video file. Returns true if everything went smoothly and false
  // otherwise.
  bool Open(const std::string& filename);

  // Reads through the media stream until the next video frame is found. Decodes
  // the video frame and returns it as an RGB image. If metadata packets are
  // encountered, the last one is retained, and can be accessed by calling
  // LastMetadataPacket().
  // If the end of the stream has been reached, an empty cv::Mat3b will be
  // returned.
  cv::Mat3b ReadFrame();

  // Returns a packet from the metadata stream, if one is available. This packet
  // will be the one that is closest in time to the last returned video frame
  // from ReadFrame(). If none exists, nullptr is returned.
  void* ReadMetadataPacket();

  Mesh* GetMesh(const std::string& key);

  // Seeks to a time in the video. Returns true if successful or false if
  // outside the bounds of the video or otherwise unsuccessful.
  bool Seek(const unsigned int time_in_ms);

  // Returns the length of the video in ms.
  unsigned int GetVideoLength() const;

  // Return the dimensions of the video frames.
  int GetWidth() const;
  int GetHeight() const;

 private:
  bool ReadUntilPacketFromStream(const int stream_index);
  bool file_open_ = false;
  unsigned int video_length_;
  unsigned int current_time_ms_;
  AVFormatContext* format_context_ = nullptr;
  AVCodecContext* video_decoder_context_ = nullptr;
  AVCodec* video_decoder_ = nullptr;
  AVStream* video_stream_ = nullptr;
  unsigned int width_, height_;
  int video_stream_idx_ = -1;
  int metadata_stream_idx_ = -1;
  AVPixelFormat pixel_format_;
  std::vector<std::pair<int, void*>> metadata_;
  std::vector<std::pair<int, cv::Mat3b>> frames_;
  int video_pts_ = 0;

  static bool registered_avcodec_;
};

}  // namespace replay
