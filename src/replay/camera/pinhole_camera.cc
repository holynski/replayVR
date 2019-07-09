#include "replay/camera/pinhole_camera.h"

#include <glog/logging.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace replay {

PinholeCamera::PinholeCamera() {
  distortion_coeffs_.resize(2, 0);
  type_ = CameraType::PINHOLE;
  intrinsics_ = Eigen::Matrix3d::Identity();
  extrinsics_ = Eigen::Matrix4d::Identity();
  image_size_ = Eigen::Vector2i(0, 0);
}

Camera* PinholeCamera::Clone() const {
  PinholeCamera* retval = new PinholeCamera();
  retval->SetIntrinsicsMatrix(GetIntrinsicsMatrix());
  retval->SetExtrinsics(GetExtrinsics());
  retval->SetDistortionCoeffs(GetDistortionCoeffs());
  retval->SetImageSize(GetImageSize());
  retval->SetName(GetName());
  return retval;
}

Eigen::Vector2d PinholeCamera::GetFOV() const {
  return Eigen::Vector2d(2 * atan2(0.5, intrinsics_(0, 0)),
                         2 * atan2(0.5, intrinsics_(1, 1)))
             .array()
             .abs() *
         (180.f / M_PI);
}

void PinholeCamera::SetFocalLengthFromFOV(const Eigen::Vector2d& fov) {
  intrinsics_(0, 0) = 1.0 / (2 * tan(fov[0] * (M_PI / 180.f) * 0.5));
  intrinsics_(1, 1) = 1.0 / (2 * tan(fov[1] * (M_PI / 180.f) * 0.5));
}

cv::Mat PinholeCamera::UndistortImage(const cv::Mat& image) const {
  LOG(FATAL) << "Function not implemented";
}

Eigen::Vector2d PinholeCamera::ProjectPoint(
    const Eigen::Vector3d& point3d) const {
  Eigen::Vector2d point2d;
  Eigen::Vector3d point3d_local = point3d - GetPosition();
  point3d_local = GetRotation() * point3d_local;
  //ProjectPoint(extrinsics_.data(), intrinsics_.data(),
               //distortion_coeffs_.data(), point3d.data(), point2d.data());
  TransformCameraToPixel(intrinsics_.data(), distortion_coeffs_.data(), point3d_local.data(), point2d.data());

  if (image_size_.norm() != 0) {
    point2d[0] *= image_size_[0];
    point2d[1] *= image_size_[1];
  }
  return point2d;
}

void PinholeCamera::SetDistortionCoeffs(const std::vector<double>& coeffs) {
  CHECK_EQ(coeffs.size(), 2);
  distortion_coeffs_ = coeffs;
}

Eigen::Vector3d PinholeCamera::PixelToWorldRay(
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

}  // namespace replay
