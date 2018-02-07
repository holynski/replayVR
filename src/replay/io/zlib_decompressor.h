#pragma once

#include <stdint.h>
#include "zlib.h"

namespace replay {

class ZlibDecompressor {
 public:
  // Constructor.
  //  has_header: does this compressed data stream have zlib headers?
  //  chunk_size: how big of a data buffer should be maintained for the
  //  decompressed data? A larger buffer is more efficient, but may not be
  //  available on certain systems.
  ZlibDecompressor(const bool has_header, const size_t chunk_size = 8196,
                   const uint32_t encoding = 0x64666c38);

  // Initializes the internal data structures for decompressing a data stream
  // (compressed_data). compressed_size refers to the number of valid bytes in
  // compressed_data. This function will return false if the decompressor can
  // not be initialized. This function needs to be called before any other
  // member function calls.
  bool Initialize(unsigned char* compressed_data, const size_t compressed_size);

  // Returns true we have read all bytes in the decompressed stream
  bool EndOfStream() const;

  // Functions for reading data from the decompressed stream. These functions
  // all advance the stream by the number of bytes read, i.e. the same data
  // cannot be read twice. Initialize() must be called before any of these
  // functions.
  unsigned char* ReadData(const size_t size);
  uint8_t ReadByte();
  uint16_t ReadShortLE();
  uint16_t ReadShortBE();
  int ReadIntLE();
  int ReadIntBE();
  uint32_t ReadUnsignedIntLE();
  uint32_t ReadUnsignedIntBE();
  float ReadFloatBE();
  float ReadFloatLE();

 private:
  bool FetchChunk();
  bool initialized_;
  z_stream stream_;
  unsigned char* compressed_data_;
  size_t compressed_size_;
  unsigned char* uncompressed_chunk_;
  size_t out_stream_position_;
  size_t bytes_read_;
  bool stream_end_;
  const bool has_header_;
  const size_t chunk_size_;
  uint32_t crc_;
};

}  // namespace replay
