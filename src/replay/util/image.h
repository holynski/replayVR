#pragma once

#include <FreeImage.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace replay {

// Reads the image file header to determine size (without reading the full image
// data)
Eigen::Vector2i GetImageSizeFromHeader(const std::string& filename);

// Bilinear pixel fetch
template <typename T>
T BilinearFetch(const cv::Mat_<T>& image, const float x, const float y) {
  T color = T();
  Eigen::Vector2i floored(x, y);
  Eigen::Vector2f frac(x - floored.x(), y - floored.y());

  for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
      Eigen::Vector2f coord = floored.cast<float>() + Eigen::Vector2f(dx, dy);
      coord.x() = std::min(coord.x(), image.cols - 1.0f);
      coord.y() = std::min(coord.y(), image.rows - 1.0f);

      float weight =
          (1.0f - std::abs(dy - frac.y())) * (1.0f - std::abs(dx - frac.x()));
      color += static_cast<T>(weight * image(coord.y(), coord.x()));
    }

  return color;
}

// Bilinear gradient fetch
template <typename T>
T BilinearGradient(const cv::Mat_<T>& image, const float x, const float y,
                   const int axis) {
  CHECK_LT(std::ceil(x), image.cols);
  CHECK_LT(std::ceil(y), image.rows);
  CHECK_LE(axis, 1);
  CHECK_GE(axis, 0);
  T gradient = T();
  Eigen::Vector2i floored(x, y);
  Eigen::Vector2f frac(x - floored.x(), y - floored.y());

  const T& base_pixel_value = image(floored.y(), floored.x());
  for (int sample = 0; sample < 2; ++sample) {
    Eigen::Vector2f coord =
        floored.cast<float>() +
        (axis == 0 ? Eigen::Vector2f(1, sample) : Eigen::Vector2f(sample, 1));
    coord.x() = std::min(coord.x(), image.cols - 1.0f);
    coord.y() = std::min(coord.y(), image.rows - 1.0f);

    float weight = (axis == 0 ? 1.0f - std::abs(sample - frac.y())
                              : 1.0f - std::abs(sample - frac.x()));
    gradient += static_cast<T>(
        weight * (image(coord.y(), coord.x()) - base_pixel_value));
  }

  return gradient;
}

}  // namespace replay
