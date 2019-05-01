#include <glog/logging.h>
#include <replay/io/write_float_image.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

namespace replay {

bool WriteFloatImage(const std::string& filename, const cv::Mat1f& image) {
  std::ofstream writer(filename, std::ios::out | std::ios::binary);
  if (!writer.is_open()) {
    LOG(ERROR) << "Could not open the depth map file: " << filename
               << " for writing.";
    return false;
  }
  int rows = image.rows;
  int cols = image.cols;
  writer.write(reinterpret_cast<const char*>(&(rows)), sizeof(int));
  writer.write(reinterpret_cast<const char*>(&(cols)), sizeof(int));
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      float depth = image(row, col);
      writer.write(reinterpret_cast<const char*>(&depth), sizeof(float));
    }
  }

  writer.close();
  return true;
}

}  // namespace replay
