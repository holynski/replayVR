#pragma once

#include <stdint.h>
#include <stdio.h>

namespace replay {

class StreamReader {
  public:
  virtual unsigned char* ReadData(const size_t size) = 0;
  virtual uint8_t ReadByte() = 0;
  virtual uint16_t ReadShortLE() = 0;
  virtual uint16_t ReadShortBE() = 0;
  virtual uint32_t ReadUnsignedIntLE() = 0;
  virtual uint32_t ReadUnsignedIntBE() = 0;
  virtual int ReadIntLE() = 0;
  virtual int ReadIntBE() = 0;
  virtual float ReadFloatLE() = 0;
  virtual float ReadFloatBE() = 0;
  virtual bool EndOfStream() const = 0;
};
}
