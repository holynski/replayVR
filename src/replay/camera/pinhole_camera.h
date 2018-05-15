#pragma once

#include "replay/camera/camera.h"

#include <Eigen/Dense>
#include "ceres/ceres.h"

namespace replay {

class PinholeCamera : public Camera {
 public:
  PinholeCamera();

  Eigen::Vector2d GetFOV() const override;
  void SetFocalLengthFromFOV(const Eigen::Vector2d& focal) override;
  Eigen::Vector2d ProjectPoint(const Eigen::Vector3d& point3d) const override;
  cv::Mat UndistortImage(const cv::Mat& image) const override;
  void SetDistortionCoeffs(const std::vector<double>& coeffs) override;

  static const int kNumDistortionCoeffs = 2;

  //
  // Templated functions that can be used with Ceres
  //

  template <typename T>
  static void DistortPoint(const T* distortion, const T* undisorted_point,
                           T* distorted_point);
  template <typename T>
  static void TransformCameraToPixel(const T* intrinsics, const T* distortion,
                                     const T* camera, T* pixel);

  template <typename T>
  static void ProjectPoint(const T* extrinsics, const T* intrinsics,
                           const T* distortion, const T* point3d, T* pixel2d);

 private:
};

template <typename T>
void PinholeCamera::TransformCameraToPixel(const T* intrinsics,
                                           const T* distortion, const T* camera,
                                           T* pixel) {
  T normalized[2];
  normalized[0] = camera[0] / camera[2];
  normalized[1] = camera[1] / camera[2];

  // Apply radial distortion.
  T distorted[2];
  PinholeCamera::DistortPoint(distortion, normalized, distorted);

  pixel[0] = intrinsics[0] * distorted[0] + intrinsics[3] * distorted[1] +
             intrinsics[6];
  pixel[1] = intrinsics[1] * distorted[1] + intrinsics[7];
}

template <typename T>
void PinholeCamera::DistortPoint(const T* distortion,
                                 const T* undistorted_point,
                                 T* distorted_point) {
  const T& k1 = distortion[0];
  const T& k2 = distortion[1];

  const T squared_radius = undistorted_point[0] * undistorted_point[0] +
                           undistorted_point[1] * undistorted_point[1];
  const T distortion_factor = 1.0 + squared_radius * (k1 + squared_radius * k2);

  distorted_point[0] = undistorted_point[0] * distortion_factor;
  distorted_point[1] = undistorted_point[1] * distortion_factor;
}

template <typename T>
void PinholeCamera::ProjectPoint(const T* extrinsics, const T* intrinsics,
                                 const T* distortion, const T* point3d,
                                 T* pixel2d) {
  T camera_point[3];
  TransformWorldToCamera(extrinsics, point3d, camera_point);
  TransformCameraToPixel(intrinsics, distortion, camera_point, pixel2d);
}

}  // namespace replay
