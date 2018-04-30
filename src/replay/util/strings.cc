#include <iostream>
#include <sstream>

namespace replay {

std::string PadZeros(const int input, const int length) {
  std::stringstream ss;
  ss.fill('0');
  ss.width(length);
  ss << input;
  return ss.str();
}

}
