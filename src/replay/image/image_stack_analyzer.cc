#include "replay/image/image_stack_analyzer.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
namespace replay {

ImageStackAnalyzer::ImageStackAnalyzer(ImageStackAnalyzer::Options options)
    : options_(options) {}

void ImageStackAnalyzer::AddImage(const cv::Mat& image, const cv::Mat1b& mask) {
  CHECK(!image.empty());
  cv::Mat img_float;
  if (image.depth() != CV_64F) {
    image.convertTo(img_float, CV_64F);
  } else {
    img_float = image.clone();
  }
  if (Ex_.empty()) {
    depth_ = image.depth();
    if (options_.compute_mean_variance) {
      Ex_ = img_float.clone();
      Ex_.setTo(0.0f);
      Ex_sq_ = Ex_.clone();
      K_ = img_float.clone();
      known_K_ = cv::Mat(img_float.size(), img_float.type(), 0.f);
      known_K_.setTo(1, mask);
      N_ = img_float.clone();
      N_.setTo(1.0, mask);
    }
    if (options_.compute_min) {
      min_ = img_float.clone();
      min_.setTo(DBL_MAX, mask == 0);
    }
    if (options_.compute_max) {
      max_ = img_float.clone();
      max_.setTo(0, mask == 0);
    }
  } else {
    if (options_.compute_mean_variance) {
      CHECK_EQ(image.size(), Ex_.size());
      CHECK_EQ(image.depth(), depth_);
      CHECK_EQ(image.channels(), Ex_.channels());
      img_float.copyTo(K_, known_K_ == 0);
      known_K_.setTo(1, mask);
      cv::add(N_, 1, N_, mask);
      cv::add(Ex_, img_float - K_, Ex_, mask);
      cv::Mat image_sq;
      cv::pow(img_float - K_, 2, image_sq);
      cv::add(Ex_sq_, image_sq, Ex_sq_, mask);
    }
    if (options_.compute_min) {
      min_.copyTo(img_float, mask == 0);
      min_ = cv::min(min_, img_float);
    }
    if (options_.compute_max) {
      max_.copyTo(img_float, mask == 0);
      max_ = cv::max(max_, img_float);
    }
  }
}

cv::Mat ImageStackAnalyzer::GetVariance() const {
  //  (Ex2 - (Ex * Ex)/n)/(n - 1)
  CHECK(options_.compute_mean_variance);
  cv::Mat sq_Ex;
  cv::pow(Ex_, 2, sq_Ex);
  cv::Mat N_positive = N_.clone();
  N_positive.setTo(1, N_positive == 0);
  cv::divide(sq_Ex, N_positive, sq_Ex);
  cv::subtract(Ex_sq_, sq_Ex, sq_Ex);
  cv::divide(sq_Ex, N_positive, sq_Ex);
  cv::Mat output;
  sq_Ex.convertTo(output, CV_32F);
  return output;
}

cv::Mat ImageStackAnalyzer::GetMean() const {
  CHECK(options_.compute_mean_variance);
  cv::Mat mean;
  cv::Mat N_positive = N_.clone();
  N_positive.setTo(1, N_ == 0);
  cv::divide(Ex_, N_positive, mean);
  cv::add(mean, K_, mean);
  cv::Mat output;
  mean.convertTo(output, depth_);
  return output;
}

cv::Mat ImageStackAnalyzer::GetMin() const {
  CHECK(options_.compute_min);
  cv::Mat output;
  min_.convertTo(output, depth_);
  return output;
}

cv::Mat ImageStackAnalyzer::GetMax() const {
  CHECK(options_.compute_max);
  cv::Mat output;
  max_.convertTo(output, depth_);
  return output;
}

cv::Mat ImageStackAnalyzer::GetMedian() const {
  LOG(FATAL) << "Function not implemented!";
  return cv::Mat();
}

}  // namespace replay

