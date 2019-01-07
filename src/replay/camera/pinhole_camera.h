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
  Eigen::Vector3d PixelToWorldRay(
      const Eigen::Vector2d& point2d) const override;

  static const int kNumDistortionCoeffs = 2;

  //
  // Templated functions that can be used with Ceres
  //

  template <typename T>
  static void DistortPoint(const T* distortion, const T* undisorted_point,
                           T* distorted_point);
  template <typename T>
  static void UndistortPoint(const T* distortion, const T* undisorted_point,
                             T* distorted_point);
  template <typename T>
  static void TransformCameraToPixel(const T* intrinsics, const T* distortion,
                                     const T* camera, T* pixel);
  template <typename T>
  static void TransformPixelToCamera(const T* intrinsics, const T* distortion,
                                     const T* pixel, T* camera);

  template <typename T>
  static void ProjectPoint(const T* extrinsics, const T* intrinsics,
                           const T* distortion, const T* point3d, T* pixel2d);
  template <typename T>
  static void UnprojectPoint(const T* extrinsics, const T* intrinsics,
                             const T* distortion, const T* point2d,
                             const T depth, const T* point3d);

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
void PinholeCamera::UndistortPoint(const T* distortion,
                                   const T* distorted_point,
                                   T* undistorted_point) {
  const T& k1 = distortion[0];
  const T& k2 = distortion[1];

  T prev_undistorted_point[2];
  undistorted_point[0] = distorted_point[0];
  undistorted_point[1] = distorted_point[1];
  // Undistort point with fisheye parameters
  //
  for (size_t i = 0; i < 100; ++i) {
    prev_undistorted_point[0] = undistorted_point[0];
    prev_undistorted_point[1] = undistorted_point[1];
    const T squared_radius = (undistorted_point[0] * undistorted_point[0] +
                 undistorted_point[1] * undistorted_point[1]);

    const T radial_distortion = 1.0 + squared_radius * (k1 + squared_radius * k2);

    undistorted_point[0] =
        distorted_point[0] / radial_distortion;
    undistorted_point[1] =
        distorted_point[1] / radial_distortion;

    // Keep doing this until we've converged.
    if (ceres::abs(undistorted_point[0] - prev_undistorted_point[0]) < 1e-10 &&
        ceres::abs(undistorted_point[1] - prev_undistorted_point[1]) < 1e-10) {
      break;
    }
  }
}

template <typename T>
void PinholeCamera::ProjectPoint(const T* extrinsics, const T* intrinsics,
                                 const T* distortion, const T* point3d,
                                 T* pixel2d) {
  T camera_point[3];
  TransformWorldToCamera(extrinsics, point3d, camera_point);
  TransformCameraToPixel(intrinsics, distortion, camera_point, pixel2d);
}

template <typename T>
void PinholeCamera::TransformPixelToCamera(const T* intrinsics,
                                           const T* distortion, const T* pixel,
                                           T* camera) {
  T distorted_point[2];
  distorted_point[1] = (pixel[1] - intrinsics[7]) / intrinsics[4];
  distorted_point[0] =
      (pixel[0] - intrinsics[6] - distorted_point[1] * intrinsics[3]) /
      intrinsics[0];

  PinholeCamera::UndistortPoint(distortion, distorted_point, camera);
  camera[2] = T(1.0);
}

}  // namespace replay
