#pragma once

#include <replay/io/stream_reader.h>
#include <cstdint>
#include <memory>

namespace replay {

class ByteStreamReader : public StreamReader {
  public:
  ByteStreamReader(const uint8_t* stream, const size_t size);
  unsigned char* ReadData(const size_t size);
  uint8_t ReadByte();
  uint16_t ReadShortLE();
  uint16_t ReadShortBE();
  uint32_t ReadUnsignedIntLE();
  uint32_t ReadUnsignedIntBE();
  int ReadIntLE();
  int ReadIntBE();
  float ReadFloatLE();
  float ReadFloatBE();
  bool EndOfStream() const;
  private:
  size_t position_;
  const uint8_t* stream_;
  const size_t stream_size_;

};

}
