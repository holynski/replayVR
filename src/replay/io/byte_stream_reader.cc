#include <replay/io/byte_stream_reader.h>
#include <cstdint>
#include <glog/logging.h>

namespace replay {

ByteStreamReader::ByteStreamReader(const uint8_t* stream, const size_t size)
    : position_(0), stream_(stream), stream_size_(size) {}

unsigned char* ByteStreamReader::ReadData(const size_t size) {
  CHECK_LE(position_ + size, stream_size_) << "Reading over end of stream.";
  unsigned char* return_value = new unsigned char[size];
  memcpy(return_value, stream_, size);
  return return_value;
}

uint8_t ByteStreamReader::ReadByte() {
  return reinterpret_cast<uint8_t*>(ReadData(1))[0];
}

uint16_t ByteStreamReader::ReadShortLE() {
  return ((static_cast<uint16_t>(ReadByte())) |
          (static_cast<uint16_t>(ReadByte()) << 8));
}

uint16_t ByteStreamReader::ReadShortBE() {
  return ((static_cast<uint16_t>(ReadByte()) << 8) |
          static_cast<uint16_t>(ReadByte()));
}

int ByteStreamReader::ReadIntLE() {
  return ((static_cast<int>(ReadByte())) | (static_cast<int>(ReadByte()) << 8) |
          (static_cast<int>(ReadByte()) << 16) |
          (static_cast<int>(ReadByte()) << 24));
}

int ByteStreamReader::ReadIntBE() {
  return ((static_cast<int>(ReadByte()) << 24) |
          (static_cast<int>(ReadByte()) << 16) |
          (static_cast<int>(ReadByte()) << 8) | static_cast<int>(ReadByte()));
}

uint32_t ByteStreamReader::ReadUnsignedIntBE() {
  return ((static_cast<uint32_t>(ReadByte()) << 24) |
          (static_cast<uint32_t>(ReadByte()) << 16) |
          (static_cast<uint32_t>(ReadByte()) << 8) |
          static_cast<uint32_t>(ReadByte()));
}

uint32_t ByteStreamReader::ReadUnsignedIntLE() {
  return ((static_cast<uint32_t>(ReadByte())) |
          (static_cast<uint32_t>(ReadByte()) << 8) |
          (static_cast<uint32_t>(ReadByte()) << 16) |
          (static_cast<uint32_t>(ReadByte()) << 24));
}

float ByteStreamReader::ReadFloatLE() {
  uint32_t return_value = ReadUnsignedIntLE();
  return reinterpret_cast<float*>(&return_value)[0];
}

float ByteStreamReader::ReadFloatBE() {
  uint32_t return_value = ReadUnsignedIntBE();
  return reinterpret_cast<float*>(&return_value)[0];
}

bool ByteStreamReader::EndOfStream() const { return position_ >= stream_size_;}
}

