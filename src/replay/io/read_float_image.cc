#include <replay/io/read_float_image.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <glog/logging.h>

namespace replay {

cv::Mat1f ReadFloatImage(const std::string& filename) {
  std::ifstream reader(filename, std::ios::in | std::ios::binary);
  if (!reader.is_open()) {
    LOG(ERROR) << "Couldn't open file: " << filename;
    return cv::Mat1f();
  }
  int cols, rows;
  reader.read(reinterpret_cast<char*>(&(rows)), sizeof(int));
  reader.read(reinterpret_cast<char*>(&(cols)), sizeof(int));
  cv::Mat1f image(rows, cols);
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      float value;
      reader.read(reinterpret_cast<char*>(&(value)), sizeof(float));
      image(row, col) = value;
    }
  }
  return image;
}

}  // namespace replay
