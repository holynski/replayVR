#pragma once

#include <Eigen/Core>
#include <cereal/access.hpp>
#include <opencv2/opencv.hpp>

#include "replay/util/row_array.h"

namespace replay {

// This class holds a depth map and confidence map as 2d arrays/images. The
// depth values are defined as the distance from the camera origin to the
// corresponding point. NOTE: This is different than the projective distance
// used in the PixelToUnitDepthRay function in the camera class.
class DepthMap {
 public:
  DepthMap() {}

  // Loads a depth map from a file. See supported filetypes in
  // io/read_depth_map.h
  explicit DepthMap(const std::string& filename);

  // Loads a depth map from the depth image, taking the pixel value as the depth
  // value. This expects a single channel image as input.
  explicit DepthMap(const cv::Mat1f& depth_image);

  // Initializes a depth map to the given size. All depth and confidence values
  // are set to zero.
  DepthMap(const int rows, const int cols);

  // Saves a depth map to file. See supported file extensions in
  // io/write_depth_map.h
  bool Save(const std::string& filename);

  // Resizes a depth map to the given size. All depth and confidence values
  // are set to zero and the previous values are NOT preserved in any way.
  void Resize(int rows, int cols);

  // Resizes the depth map and confidence values, interpolates using
  // cv::INTER_NEAREST
  void ResizeAndInterpolate(int rows, int cols);

  // Return the size of the depth map.
  int Rows() const;
  int Cols() const;
  int Height() const;
  int Width() const;

  // Set and get the depth value at the given pixel location.
  void SetDepth(int row, int col, const float depth);
  float GetDepth(int row, int col) const;
  float BilinearInterpolateDepth(double row, double col) const;

  // Set and get the confidence value at the given pixel location.
  void SetConfidence(int row, int col, const float confidence);
  float GetConfidence(int row, int col) const;
  float BilinearInterpolateConfidence(double row, double col) const;

  // Direct accessors to the underlying depth map. Note that these methods
  // should never alter the size of the returned arrays.
  cv::Mat1f* MutableDepth();
  const cv::Mat1f& Depth() const;

  // Direct accessors to the underlying confidence map. Note that these methods
  // should never alter the size of the returned arrays.
  cv::Mat1f* MutableConfidence();
  const cv::Mat1f& Confidence() const;

  // Writes a RGB image that represents the depth map. The RGB values are based
  // on an interpolation scheme where blue (i.e., "cold") means lower depth
  // values and red (i.e. "warm") means higher depth values.
  // If min_depth >= max_depth, the values are populated automatically with the
  // min and max depth in the image.
  void WriteDepthAsRGB(const std::string& file, float min_depth = -1,
                            float max_depth = -1) const;

  // Same as WriteDepthAsRGBImage, but using a 0-255 grayscale image.
  void WriteDepthAsGrayscale(const std::string& output_file,
                             float min_depth = -1,
                             float max_depth = -1) const;

  void WriteConfidenceAsGrayscale(const std::string& output_file) const;

 private:
  cv::Mat1f depth_;
  cv::Mat1f confidence_;
};

}  // namespace replay
