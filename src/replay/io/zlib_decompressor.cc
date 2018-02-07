#include "zlib_decompressor.h"
#include <glog/logging.h>
#include <math.h>
#include <stdlib.h>
#include "zlib.h"

namespace replay {

int called_fetch_chunk_ = 0;
ZlibDecompressor::ZlibDecompressor(const bool has_header,
                                   const size_t chunk_size,
                                   const uint32_t encoding)
    : initialized_(false),
      uncompressed_chunk_(nullptr),
      out_stream_position_(0),
      bytes_read_(0),
      stream_end_(false),
      has_header_(has_header),
      chunk_size_(chunk_size),
      crc_(crc32(0, Z_NULL, 0)) {
  crc_ = crc32(crc_, reinterpret_cast<const uint8_t*>(&encoding), 4);
  stream_.zalloc = Z_NULL;
  stream_.zfree = Z_NULL;
  stream_.opaque = Z_NULL;
  stream_.avail_out = chunk_size_;
  uncompressed_chunk_ = new uint8_t[chunk_size_];
  stream_.next_out = uncompressed_chunk_;
}

bool ZlibDecompressor::FetchChunk() {
  called_fetch_chunk_++;
  if (stream_end_) {
    return false;
  }
  stream_.avail_out = chunk_size_;
  stream_.next_out = uncompressed_chunk_;
  int return_code = 0;
  while (stream_.avail_out > 0) {
    return_code = inflate(&stream_, Z_NO_FLUSH);
    if (return_code < 0) {
      LOG(INFO) << "ERROR: " << return_code;
      return false;
    }
    if (return_code == Z_STREAM_END) {
      stream_end_ = true;
      break;
    }
  }
  out_stream_position_ = 0;
  if (stream_end_) {
    LOG(INFO) << crc_;
  }
  return true;
}

bool ZlibDecompressor::Initialize(unsigned char* compressed_data,
                                  const size_t compressed_size) {
  crc_ = crc32(crc_, compressed_data, compressed_size);
  compressed_data_ = compressed_data;
  compressed_size_ = compressed_size;
  stream_.avail_in = compressed_size;
  stream_.next_in = compressed_data_;
  if (has_header_) {
    if (inflateInit(&stream_) != Z_OK) {
      return false;
    }
  } else {
    if (inflateInit2(&stream_, -15) != Z_OK) {
      return false;
    }
  }
  FetchChunk();
  initialized_ = true;
  return true;
}

bool ZlibDecompressor::EndOfStream() const { return stream_end_; }

unsigned char* ZlibDecompressor::ReadData(const size_t size) {
  CHECK(initialized_) << "Call Initialize() first!";
  unsigned char* output_stream;
  output_stream = new uint8_t[size];
  size_t saved_bytes = 0;
  while (saved_bytes < size) {
    int num_bytes_to_copy =
        std::min(chunk_size_ - out_stream_position_, size - saved_bytes);
    if (num_bytes_to_copy > 0) {
      memcpy(output_stream + saved_bytes,
             uncompressed_chunk_ + out_stream_position_, num_bytes_to_copy);
    }
    out_stream_position_ += num_bytes_to_copy;
    if (num_bytes_to_copy == 0) {
      if (!FetchChunk()) {
        break;
      }
    }
    saved_bytes += num_bytes_to_copy;
  }

  bytes_read_ += size;
  return output_stream;
}

uint8_t ZlibDecompressor::ReadByte() {
  return reinterpret_cast<uint8_t*>(ReadData(1))[0];
}

uint16_t ZlibDecompressor::ReadShortLE() {
  return ((static_cast<uint16_t>(ReadByte())) |
          (static_cast<uint16_t>(ReadByte()) << 8));
}

uint16_t ZlibDecompressor::ReadShortBE() {
  return ((static_cast<uint16_t>(ReadByte()) << 8) |
          static_cast<uint16_t>(ReadByte()));
}

int ZlibDecompressor::ReadIntLE() {
  return ((static_cast<int>(ReadByte())) | (static_cast<int>(ReadByte()) << 8) |
          (static_cast<int>(ReadByte()) << 16) |
          (static_cast<int>(ReadByte()) << 24));
}

int ZlibDecompressor::ReadIntBE() {
  return ((static_cast<int>(ReadByte()) << 24) |
          (static_cast<int>(ReadByte()) << 16) |
          (static_cast<int>(ReadByte()) << 8) | static_cast<int>(ReadByte()));
}

unsigned int reverseBits(uint32_t num) {
  size_t NO_OF_BITS = sizeof(num) * 8;
  uint32_t reverse_num = 0;
  size_t i;
  for (i = 0; i < NO_OF_BITS; i++) {
    if ((num & (1 << i))) reverse_num |= 1 << ((NO_OF_BITS - 1) - i);
  }
  return reverse_num;
}

uint32_t ZlibDecompressor::ReadUnsignedIntBE() {
  return ((static_cast<uint32_t>(ReadByte()) << 24) |
          (static_cast<uint32_t>(ReadByte()) << 16) |
          (static_cast<uint32_t>(ReadByte()) << 8) |
          static_cast<uint32_t>(ReadByte()));
}

uint32_t ZlibDecompressor::ReadUnsignedIntLE() {
  return ((static_cast<uint32_t>(ReadByte())) |
          (static_cast<uint32_t>(ReadByte()) << 8) |
          (static_cast<uint32_t>(ReadByte()) << 16) |
          (static_cast<uint32_t>(ReadByte()) << 24));
}

float ZlibDecompressor::ReadFloatLE() {
  uint32_t return_value = ReadUnsignedIntLE();
  return reinterpret_cast<float*>(&return_value)[0];
}

float ZlibDecompressor::ReadFloatBE() {
  uint32_t return_value = ReadUnsignedIntBE();
  return reinterpret_cast<float*>(&return_value)[0];
}

}  // namespace replay
