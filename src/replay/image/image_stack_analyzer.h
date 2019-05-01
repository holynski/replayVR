#pragma once

#include <opencv2/opencv.hpp>

namespace replay {

class ImageStackAnalyzer {
 public:
  struct Options {
    inline Options() {}
    bool compute_mean_variance = true;
    bool compute_min = true;
    bool compute_max = true;
    bool compute_median = true;
  };

  ImageStackAnalyzer(Options options = Options());
  void AddImage(const cv::Mat& image, const cv::Mat1b& mask);
  cv::Mat GetVariance() const;
  cv::Mat GetMean() const;
  cv::Mat GetMedian() const;
  cv::Mat GetMin() const;
  cv::Mat GetMax() const;

 private:
  // The sum per pixel
  cv::Mat Ex_;
  // The squared sum per pixel
  cv::Mat Ex_sq_;
  // The mean-shift per pixel
  cv::Mat K_;
  // The pixels of K_ which have been filled
  cv::Mat known_K_;
  // Number of samples per pixel
  cv::Mat N_;
  // Minimum value
  cv::Mat min_;
  // Maximum value
  cv::Mat max_;

  int depth_;

  const Options options_;
};

}  // namespace replay
