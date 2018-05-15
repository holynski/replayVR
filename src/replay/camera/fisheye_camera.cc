#include "replay/camera/fisheye_camera.h"

#include <glog/logging.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace replay {

FisheyeCamera::FisheyeCamera() {
  distortion_coeffs_.resize(4, 0);
  type_ = CameraType::FISHEYE;
  intrinsics_ = Eigen::Matrix3d::Identity();
  extrinsics_ = Eigen::Matrix4d::Identity();
  image_size_ = Eigen::Vector2i(0, 0);
}

Eigen::Vector2d FisheyeCamera::GetFOV() const {
  return Eigen::Vector2d(2 * atan2(image_size_[0] * 0.5, intrinsics_(0, 0)),
                         2 * atan2(image_size_[1] * 0.5, intrinsics_(1, 1)))
             .array()
             .abs() *
         (180.f / M_PI);
}

void FisheyeCamera::SetFocalLengthFromFOV(const Eigen::Vector2d& fov) {
  CHECK_GE(image_size_[0], 0);
  CHECK_GE(image_size_[1], 0);
  intrinsics_(0, 0) = image_size_[0] / (fov[0] * M_PI / 180.f);
  intrinsics_(1, 1) = image_size_[1] / (fov[1] * M_PI / 180.f);
}

Eigen::Vector2d FisheyeCamera::ProjectPoint(
    const Eigen::Vector3d& point3d) const {
  Eigen::Vector2d point2d;
  ProjectPoint(extrinsics_.data(), intrinsics_.data(),
               distortion_coeffs_.data(), point3d.data(), point2d.data());
  return point2d;
}

void FisheyeCamera::SetDistortionCoeffs(const std::vector<double>& coeffs) {
  CHECK_EQ(coeffs.size(), 4);
  distortion_coeffs_ = coeffs;
}

cv::Mat FisheyeCamera::UndistortImage(const cv::Mat& image) const {
  LOG(FATAL) << "Function not implemented";
}

}  // namespace replay
