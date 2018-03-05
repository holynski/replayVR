#include "replay/depth_map/depth_map.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <replay/third_party/theia/math/util.h>

#include "replay/util/row_array.h"

namespace replay {
namespace {
Eigen::Vector3d GetRGBFromGray(float v_in, float vmin, float vmax) {
  Eigen::Vector3d c(1.0, 1.0, 1.0);

  const float v = theia::Clamp(v_in, vmin, vmax);
  const float dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv)) {
    c(0) = 0.0;
    c(1) = 4.0 * (v - vmin) / dv;
  } else if (v < (vmin + 0.5 * dv)) {
    c(0) = 0.0;
    c(2) = 1.0 + 4.0 * (vmin + 0.25 * dv - v) / dv;
  } else if (v < (vmin + 0.75 * dv)) {
    c(0) = 4.0 * (v - vmin - 0.5 * dv) / dv;
    c(2) = 0.0;
  } else {
    c(1) = 1.0 + 4.0 * (vmin + 0.75 * dv - v) / dv;
    c(2) = 0;
  }

  return c;
}
}  // namespace

DepthMap::DepthMap(const std::string& filename) {
  LOG(FATAL) << "Not implemented";
}

DepthMap::DepthMap(const cv::Mat1f& image) {
  CHECK_EQ(image.channels(), 1);

  depth_ = image.clone();
}

bool DepthMap::Save(const std::string& filename) {
  LOG(FATAL) << "Not implemented.";
}

DepthMap::DepthMap(const int rows, const int cols) {
  depth_ = cv::Mat1f(rows, cols, 0.0f);
  confidence_ = cv::Mat1f(rows, cols, 0.0f);
}

void DepthMap::Resize(int rows, int cols) {
  depth_ = cv::Mat1f(rows, cols, 0.0f);
  confidence_ = cv::Mat1f(rows, cols, 0.0f);
}

void DepthMap::ResizeAndInterpolate(int rows, int cols) {
  cv::resize(depth_, depth_, cv::Size(cols, rows), 0, 0, cv::INTER_NEAREST);
  cv::resize(confidence_, confidence_, cv::Size(cols, rows), 0, 0,
             cv::INTER_NEAREST);
}

int DepthMap::Rows() const { return depth_.rows; }

int DepthMap::Cols() const { return depth_.cols; }

int DepthMap::Width() const { return depth_.cols; }

int DepthMap::Height() const { return depth_.rows; }

void DepthMap::SetDepth(int row, int col, const float depth) {
  DCHECK_GE(row, 0);
  DCHECK_LT(row, depth_.rows);
  DCHECK_GE(col, 0);
  DCHECK_LT(col, depth_.cols);
  depth_(row, col) = depth;
}

float DepthMap::GetDepth(int row, int col) const {
  DCHECK_GE(row, 0);
  DCHECK_LT(row, depth_.rows);
  DCHECK_GE(col, 0);
  DCHECK_LT(col, depth_.cols);
  return depth_(row, col);
}

float DepthMap::BilinearInterpolateDepth(double row, double col) const {
  DCHECK_GE(row, 0);
  DCHECK_LE(row, depth_.rows - 1);
  DCHECK_GE(col, 0);
  DCHECK_LE(col, depth_.cols - 1);

  const int col_left = static_cast<int>(col);
  const int row_top = static_cast<int>(row);
  const double dcol = col - col_left;
  const double drow = row - row_top;

  return (1.0 - dcol) * (1.0 - drow) * GetDepth(row_top, col_left) +
         dcol * (1.0 - drow) * GetDepth(row_top, col_left + 1) +
         (1.0 - dcol) * drow * GetDepth(row_top + 1, col_left) +
         dcol * drow * GetDepth(row_top + 1, col_left + 1);
}

void DepthMap::SetConfidence(int row, int col, const float confidence) {
  DCHECK_GE(row, 0);
  DCHECK_LT(row, confidence_.rows);
  DCHECK_GE(col, 0);
  DCHECK_LT(col, confidence_.cols);
  confidence_(row, col) = confidence;
}

float DepthMap::GetConfidence(int row, int col) const {
  DCHECK_GE(row, 0);
  DCHECK_LT(row, confidence_.rows);
  DCHECK_GE(col, 0);
  DCHECK_LT(col, confidence_.cols);
  return confidence_(row, col);
}

float DepthMap::BilinearInterpolateConfidence(double row, double col) const {
  DCHECK_GE(row, 0);
  DCHECK_LE(row, confidence_.rows - 1);
  DCHECK_GE(col, 0);
  DCHECK_LE(col, confidence_.cols - 1);

  const int col_left = static_cast<int>(col);
  const int row_top = static_cast<int>(row);
  const double dcol = col - col_left;
  const double drow = row - row_top;

  return (1.0 - dcol) * (1.0 - drow) * GetConfidence(row_top, col_left) +
         dcol * (1.0 - drow) * GetConfidence(row_top, col_left + 1) +
         (1.0 - dcol) * drow * GetConfidence(row_top + 1, col_left) +
         dcol * drow * GetConfidence(row_top + 1, col_left + 1);
}

cv::Mat1f* DepthMap::MutableDepth() { return &depth_; }

const cv::Mat1f& DepthMap::Depth() const { return depth_; }

cv::Mat1f* DepthMap::MutableConfidence() { return &confidence_; }

const cv::Mat1f& DepthMap::Confidence() const { return confidence_; }

void DepthMap::WriteDepthAsRGB(const std::string& output_file,
                                    float min_depth, float max_depth) const {
  cv::Mat3b rgb(depth_.cols, depth_.rows, 3);

  if (max_depth <= min_depth) {
    double min, max;
    cv::minMaxLoc(depth_, &min, &max);
    min_depth = static_cast<float>(min);
    max_depth = static_cast<float>(max);
  }

  // Iterate over the pixels and convert each depth value to the RGB value.
  for (int i = 0; i < depth_.rows * depth_.cols; i++) {
    // Extract color given the depth value.
    const Eigen::Vector3d color =
        GetRGBFromGray(depth_(i), min_depth, max_depth);
    rgb(i)[0] = color(2);
    rgb(i)[1] = color(1);
    rgb(i)[2] = color(0);
  }

  cv::imwrite(output_file, rgb);
}

void DepthMap::WriteDepthAsGrayscale(const std::string& output_file,
                                          float min_depth,
                                          float max_depth) const {
  if (max_depth <= min_depth) {
    double min, max;
    cv::minMaxLoc(depth_, &min, &max);
    min_depth = static_cast<float>(min);
    max_depth = static_cast<float>(max);
  }

  cv::Mat1f normalized_depth =
      (depth_ - min_depth) / max_depth;
  normalized_depth.setTo(1.0f, normalized_depth > 1.0f);
  normalized_depth.setTo(0.0f, normalized_depth < 0.0f);

  cv::imwrite(output_file, normalized_depth * 255);
}

void DepthMap::WriteConfidenceAsGrayscale(
    const std::string& output_file) const {
  cv::imwrite(output_file, confidence_ * 255);
}

}  // namespace replay
