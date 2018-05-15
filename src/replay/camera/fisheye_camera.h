#pragma once

#include "replay/camera/camera.h"

#include <Eigen/Dense>
#include "ceres/ceres.h"

namespace replay {

class FisheyeCamera : public Camera {
 public:
  FisheyeCamera();

  Eigen::Vector2d GetFOV() const override;
  void SetFocalLengthFromFOV(const Eigen::Vector2d& focal) override;
  Eigen::Vector2d ProjectPoint(const Eigen::Vector3d& point3d) const override;
  cv::Mat UndistortImage(const cv::Mat& image) const override;
  void SetDistortionCoeffs(const std::vector<double>& coeffs) override;

  static const int kNumDistortionCoeffs = 4;

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
void FisheyeCamera::DistortPoint(const T* distortion,
                                 const T* undistorted_point,
                                 T* distorted_point) {
  // Distort point with fisheye parameters
  //
  static const T epsilon = T(1e-8);
  const T squared_radius = T(undistorted_point[0]) * T(undistorted_point[0]) +
                           T(undistorted_point[1]) * T(undistorted_point[1]);
  if (squared_radius < epsilon) {
    distorted_point[0] = T(undistorted_point[0]);
    distorted_point[1] = T(undistorted_point[1]);
  } else {
    const T num = ceres::sqrt(squared_radius);
    const T theta = ceres::atan2(num, ceres::abs(T(undistorted_point[2])));
    const T theta_sq = theta * theta;
    const T theta_d =
        theta *
        (1.0 + distortion[0] * theta_sq + distortion[1] * theta_sq * theta_sq +
         distortion[2] * theta_sq * theta_sq * theta_sq +
         distortion[3] * theta_sq * theta_sq * theta_sq * theta_sq);

    distorted_point[0] = theta_d * undistorted_point[0] / num;
    distorted_point[1] = theta_d * undistorted_point[1] / num;

    if (undistorted_point[2] < T(0.0)) {
      distorted_point[0] = -distorted_point[0];
      distorted_point[1] = -distorted_point[1];
    }
  }
}

template <typename T>
void FisheyeCamera::TransformCameraToPixel(const T* intrinsics,
                                           const T* distortion, const T* camera,
                                           T* pixel) {
  FisheyeCamera::DistortPoint(distortion, camera, pixel);
  pixel[0] =
      intrinsics[0] * pixel[0] + intrinsics[3] * pixel[1] + intrinsics[6];
  pixel[1] = intrinsics[4] * pixel[1] + intrinsics[7];
}

template <typename T>
void FisheyeCamera::ProjectPoint(const T* extrinsics, const T* intrinsics,
                                 const T* distortion, const T* point3d,
                                 T* pixel2d) {
  T camera_point[3];
  TransformWorldToCamera(extrinsics, point3d, camera_point);
  TransformCameraToPixel(intrinsics, distortion, camera_point, pixel2d);
}

}  // namespace replay
