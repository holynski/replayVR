#include "replay/camera/camera.h"

#include <glog/logging.h>
#include <Eigen/Dense>

namespace replay {

Camera::Camera() : type_(CameraType::PINHOLE) {}

CameraType Camera::GetType() const { return type_; }

Eigen::Vector2d Camera::GetFocalLength() const {
  return Eigen::Vector2d(intrinsics_(0, 0), intrinsics_(1, 1));
}

Eigen::Vector2d Camera::GetPrincipalPoint() const {
  return Eigen::Vector2d(intrinsics_(0, 2), intrinsics_(1, 2));
}

double Camera::GetSkew() const { return intrinsics_(0, 1); }

const Eigen::Matrix3d& Camera::GetIntrinsicsMatrix() const {
  return intrinsics_;
}

Eigen::Vector2i Camera::GetImageSize() const { return image_size_; }

const std::vector<double>& Camera::GetDistortionCoeffs() const {
  return distortion_coeffs_;
}

void Camera::SetFocalLength(const Eigen::Vector2d& focal) {
  intrinsics_(0, 0) = focal[0];
  intrinsics_(1, 1) = focal[1];
}

void Camera::SetPrincipalPoint(const Eigen::Vector2d& principal) {
  intrinsics_(0, 2) = principal[0];
  intrinsics_(1, 2) = principal[1];
}

void Camera::SetSkew(const double skew) { intrinsics_(0, 1) = skew; }

void Camera::SetIntrinsicsMatrix(const Eigen::Matrix3d& intrinsics) {
  intrinsics_ = intrinsics;
}

void Camera::SetImageSize(const Eigen::Vector2i& image_size) {
  image_size_ = image_size;
}

Eigen::Matrix3d Camera::GetRotation() const {
  return extrinsics_.block(0, 0, 3, 3);
}

Eigen::Vector3d Camera::GetTranslation() const {
  return extrinsics_.block(0, 3, 1, 3);
}

Eigen::Quaterniond Camera::GetOrientation() const {
  return Eigen::Quaterniond(Eigen::Matrix3d(extrinsics_.block(0, 0, 3, 3)));
}

Eigen::Vector3d Camera::GetLookAt() const {
  return extrinsics_.block(0, 0, 3, 3).row(2);
}

Eigen::Vector3d Camera::GetUpVector() const {
  return extrinsics_.block(0, 0, 3, 3).row(1);
}

Eigen::Vector3d Camera::GetRightVector() const {
  return extrinsics_.block(0, 0, 3, 3).row(0);
}

Eigen::Vector3d Camera::GetPosition() const {
  return -extrinsics_.block(0, 0, 3, 3) * extrinsics_.block(0, 3, 3, 1);
}

Eigen::Matrix4d Camera::GetExtrinsics() const { return extrinsics_; }

void Camera::SetRotation(const Eigen::Matrix3d& rotation) {
  extrinsics_.block(0, 0, 3, 3) = rotation;
}

void Camera::SetOrientation(const Eigen::Quaterniond& orientation) {
  extrinsics_.block(0, 0, 3, 3) = orientation.toRotationMatrix();
}

void Camera::SetTranslation(const Eigen::Vector3d& translation) {
  extrinsics_.block(0, 3, 1, 3) = translation;
}

void Camera::SetOrientationFromLookAtUpVector(const Eigen::Vector3d& lookat,
                                              const Eigen::Vector3d& up) {
  extrinsics_.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
  extrinsics_.block(0, 0, 3, 3).row(2) = lookat.normalized();
  extrinsics_.block(0, 0, 3, 3).row(1) = up.normalized();
}

void Camera::SetPosition(const Eigen::Vector3d& position) {
  extrinsics_.block(0, 3, 3, 1) =
      (-extrinsics_.block(0, 0, 3, 3).transpose() * position);
}

void Camera::SetExtrinsics(const Eigen::Matrix4d& extrinsics) {
  extrinsics_ = extrinsics;
}

Eigen::Matrix4f Camera::GetOpenGlProjection(double near_clip,
                                            double far_clip) const {
  Eigen::Matrix4d projection = Eigen::Matrix4d::Zero();
  projection(0, 0) = -intrinsics_(0, 0) / intrinsics_(0, 2);
  projection(1, 1) = -intrinsics_(1, 1) / intrinsics_(1, 2);
  projection(2, 2) = -(far_clip + near_clip) / (far_clip - near_clip);
  projection(2, 3) = -(2 * far_clip * near_clip) / (far_clip - near_clip);
  projection(3, 2) = -1;
  return projection.cast<float>();
}

Eigen::Matrix4f Camera::GetOpenGlExtrinsics() const {
  Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Zero();
  Eigen::Matrix3d rotation = GetRotation();
  rotation.row(0) = -rotation.row(0);
  rotation.row(2) = -rotation.row(2);
  extrinsics.block<3, 3>(0, 0) = rotation;
  extrinsics.block<3, 1>(0, 3) = -rotation * GetPosition();
  extrinsics(3, 3) = 1;
  return extrinsics.cast<float>();
}

Eigen::Matrix4f Camera::GetOpenGlMvpMatrix() const {
  return GetOpenGlProjection() * GetOpenGlExtrinsics();
}

Eigen::Vector3d Camera::UnprojectPoint(const Eigen::Vector2d& point2d,
                                       const double depth) const {
    const Eigen::Vector3d position = GetPosition();
    const Eigen::Vector3d direction = PixelToWorldRay(point2d);
    
  return position + direction.normalized() * depth;
}

double Camera::AngleFromOpticalAxis(const Eigen::Vector3d& point) const {
  const Eigen::Vector3d camera_to_point = (point - GetPosition()).normalized();
  const Eigen::Vector3d optical_axis = GetLookAt().normalized();

  return 180.0 * acos(camera_to_point.dot(optical_axis)) / M_PI;
}

const double* Camera::extrinsics() const { return extrinsics_.data(); }
double* Camera::mutable_extrinsics() { return extrinsics_.data(); }
const double* Camera::intrinsics() const { return intrinsics_.data(); }
double* Camera::mutable_intrinsics() { return intrinsics_.data(); }
const double* Camera::distortion_coeffs() const {
  return distortion_coeffs_.data();
}
double* Camera::mutable_distortion_coeffs() {
  return distortion_coeffs_.data();
}

std::string Camera::TypeToString(const CameraType type) {
  switch (type) {
    case CameraType::FISHEYE:
      return "FISHEYE";
    case CameraType::PINHOLE:
      return "PINHOLE";
    default:
      return "UNKNOWN";
  }
}

}  // namespace replay
