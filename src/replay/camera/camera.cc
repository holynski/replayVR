#include "replay/camera/camera.h"

#include <Eigen/Dense>

namespace replay {

Camera::Camera() : type_(CameraType::PINHOLE) {}

Camera::Camera(const CameraType type)
    : type_(type),
      intrinsics_(Eigen::Matrix3f::Identity()),
      extrinsics_(Eigen::Matrix4f::Identity()) {}

Eigen::Vector2f Camera::GetFocalLength() const {
  return Eigen::Vector2f(intrinsics_(0, 0), intrinsics_(1, 1));
}

Eigen::Vector2f Camera::GetPrincipalPoint() const {
  return Eigen::Vector2f(intrinsics_(0, 2), intrinsics_(1, 2));
}

Eigen::Matrix3f Camera::GetIntrinsics() const { return intrinsics_; }

Eigen::Vector2f Camera::GetFOV() const {
  return Eigen::Vector2f(2 * atan2(image_size_[0] * 0.5, intrinsics_(0, 0)),
                         2 * atan2(image_size_[1] * 0.5, intrinsics_(1, 1)))
             .array()
             .abs() *
         (180.f / M_PI);
}

Eigen::Vector2i Camera::GetImageSize() const { return image_size_; }

void Camera::SetFocalLength(const Eigen::Vector2f& focal) {
  intrinsics_(0, 0) = focal[0];
  intrinsics_(1, 1) = focal[1];
}

void Camera::SetFocalLengthFromFOV(const Eigen::Vector2f& fov) {
  intrinsics_(0, 0) = image_size_[0] / (2 * tan(fov[0] * (M_PI / 180.f) * 0.5));
  intrinsics_(1, 1) = image_size_[1] / (2 * tan(fov[1] * (M_PI / 180.f) * 0.5));
}

void Camera::SetPrincipalPoint(const Eigen::Vector2f& principal) {
  intrinsics_(0, 2) = principal[0];
  intrinsics_(1, 2) = principal[1];
}

void Camera::SetIntrinsics(const Eigen::Matrix3f& intrinsics) {
  intrinsics_ = intrinsics;
}

void Camera::SetImageSize(const Eigen::Vector2i& image_size) {
  image_size_ = image_size;
}

Eigen::Matrix3f Camera::GetRotation() const {
  return extrinsics_.block(0, 0, 3, 3);
}

Eigen::Quaternionf Camera::GetOrientation() const {
  return Eigen::Quaternionf(Eigen::Matrix3f(extrinsics_.block(0, 0, 3, 3)));
}

Eigen::Vector3f Camera::GetPosition() const {
  return -extrinsics_.block(0, 0, 3, 3) * extrinsics_.block(0, 3, 3, 1);
}

Eigen::Matrix4f Camera::GetExtrinsics() const { return extrinsics_; }

void Camera::SetRotation(const Eigen::Matrix3f& rotation) {
  extrinsics_.block(0, 0, 3, 3) = rotation;
}

void Camera::SetOrientation(const Eigen::Quaternionf& orientation) {
  extrinsics_.block(0, 0, 3, 3) = orientation.toRotationMatrix();
}

void Camera::SetOrientationFromLookAtUpVector(const Eigen::Vector3f& lookat,
                                              const Eigen::Vector3f& up) {
  extrinsics_.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();
  extrinsics_.block(0, 0, 3, 3).col(2) = lookat;
  extrinsics_.block(0, 0, 3, 3).col(1) = up;
}

void Camera::SetPosition(const Eigen::Vector3f& position) {
  extrinsics_.block(0, 3, 3, 1) = position;
}

void Camera::SetExtrinsics(const Eigen::Matrix4f& extrinsics) {
  extrinsics_ = extrinsics;
}

Eigen::Matrix4f Camera::GetOpenGlProjection(float near_clip,
                                            float far_clip) const {
  Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
  projection(0, 0) = -intrinsics_(0, 0) / intrinsics_(0, 2);
  projection(1, 1) = -intrinsics_(1, 1) / intrinsics_(1, 2);
  projection(2, 2) = -(far_clip + near_clip) / (far_clip - near_clip);
  projection(2, 3) = -(2 * far_clip * near_clip) / (far_clip - near_clip);
  projection(3, 2) = -1;
  return projection;
}

Eigen::Matrix4f Camera::GetOpenGlExtrinsics() const {
  Eigen::Matrix4f extrinsics = Eigen::Matrix4f::Zero();
  Eigen::Matrix3f rotation = GetRotation();
  rotation.row(0) = -rotation.row(0);
  rotation.row(2) = -rotation.row(2);
  extrinsics.block<3, 3>(0, 0) = rotation;
  extrinsics.block<3, 1>(0, 3) = -rotation * GetPosition();
  extrinsics(3, 3) = 1;
  return extrinsics;
}

Eigen::Matrix4f Camera::GetOpenGlMvpMatrix() const {
  return GetOpenGlProjection() * GetOpenGlExtrinsics();
}

}  // namespace replay
