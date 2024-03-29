#include "replay/camera/fisheye_camera.h"

#include <glog/logging.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace replay {

FisheyeCamera::FisheyeCamera() {
  distortion_coeffs_.resize(FisheyeCamera::kNumDistortionCoeffs, 0);
  type_ = CameraType::FISHEYE;
  intrinsics_ = Eigen::Matrix3d::Identity();
  extrinsics_ = Eigen::Matrix4d::Identity();
  image_size_ = Eigen::Vector2i(0, 0);
}

Camera* FisheyeCamera::Clone() const {
  FisheyeCamera* retval = new FisheyeCamera();
  retval->SetIntrinsicsMatrix(GetIntrinsicsMatrix());
  retval->SetExtrinsics(GetExtrinsics());
  retval->SetDistortionCoeffs(GetDistortionCoeffs());
  retval->SetImageSize(GetImageSize());
  retval->SetName(GetName());
  return retval;
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
  CHECK_EQ(coeffs.size(), FisheyeCamera::kNumDistortionCoeffs);
  distortion_coeffs_ = coeffs;
}

cv::Mat FisheyeCamera::UndistortImage(const cv::Mat& image) const {
  LOG(FATAL) << "Function not implemented";
}

Eigen::Vector3d FisheyeCamera::PixelToWorldRay(
    const Eigen::Vector2d& point2d) const {
  Eigen::Vector2d point_normalized = point2d;
  if (image_size_.norm() != 0) {
    point_normalized[0] /= image_size_[0];
    point_normalized[1] /= image_size_[1];
  }
  CHECK_LE(point_normalized.x(), 1);
  CHECK_GE(point_normalized.x(), 0);
  CHECK_LE(point_normalized.y(), 1);
  CHECK_GE(point_normalized.y(), 0);
  Eigen::Vector3d camera_ray;
  TransformPixelToCamera(intrinsics_.data(), distortion_coeffs_.data(),
                         point_normalized.data(), camera_ray.data());

  camera_ray.normalize();
  return GetRotation().transpose() * camera_ray;
}

// Need to implement undistort

}  // namespace replay
