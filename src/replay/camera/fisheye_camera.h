#pragma once

#include "replay/camera/camera.h"

#include <Eigen/Dense>
#include "ceres/ceres.h"

namespace replay {

class FisheyeCamera : public Camera {
 public:
  FisheyeCamera();
  Camera* Clone() const override;
  Eigen::Vector2d GetFOV() const override;
  void SetFocalLengthFromFOV(const Eigen::Vector2d& focal) override;
  Eigen::Vector2d ProjectPoint(const Eigen::Vector3d& point3d) const override;
  cv::Mat UndistortImage(const cv::Mat& image) const override;
  void SetDistortionCoeffs(const std::vector<double>& coeffs) override;
  Eigen::Vector3d PixelToWorldRay(
      const Eigen::Vector2d& point2d) const override;

  static const int kNumDistortionCoeffs = 5;

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
void FisheyeCamera::DistortPoint(const T* distortion,
                                 const T* undistorted_point,
                                 T* distorted_point) {
  // Distort point with fisheye parameters
  //
  static const T epsilon = T(1e-8);
  const T w = sqrt(undistorted_point[0] * undistorted_point[0] +
                   undistorted_point[1] * undistorted_point[1]);
  if (w < epsilon) {
    distorted_point[0] = T(undistorted_point[0]);
    distorted_point[1] = T(undistorted_point[1]);
  } else {
    const T theta = ceres::atan2(w, undistorted_point[2]);

    // Project the points
    T dx = theta * undistorted_point[0] / w;
    T dy = theta * undistorted_point[1] / w;

    const T sq_r = (dx * dx) + (dy * dy);
    const T radial_distortion = T(1) + distortion[0] * sq_r +
                                distortion[1] * sq_r * sq_r +
                                distortion[2] * sq_r * sq_r * sq_r;

    // Apply radial and tangential distortion
    distorted_point[0] = dx * radial_distortion +
                         T(2) * distortion[3] * dx * dy +
                         distortion[4] * (sq_r + T(2.0) * dx * dx);
    distorted_point[1] = dy * radial_distortion +
                         T(2) * distortion[4] * dx * dy +
                         distortion[3] * (sq_r + T(2.0) * dy * dy);
  }
}

template <typename T>
void FisheyeCamera::UndistortPoint(const T* distortion,
                                   const T* distorted_point,
                                   T* undistorted_point) {
  T prev_undistorted_point[2];
  undistorted_point[0] = distorted_point[0];
  undistorted_point[1] = distorted_point[1];
  // Undistort point with fisheye parameters
  //
  static const T epsilon = T(1e-8);
  for (size_t i = 0; i < 100; ++i) {
    prev_undistorted_point[0] = undistorted_point[0];
    prev_undistorted_point[1] = undistorted_point[1];
    const T w = sqrt(undistorted_point[0] * undistorted_point[0] +
                     undistorted_point[1] * undistorted_point[1]);
    if (w < epsilon) {
      undistorted_point[0] = T(distorted_point[0]);
      undistorted_point[1] = T(distorted_point[1]);
      break;
    } else {
      const T theta = ceres::atan2(w, T(1.0));

      T dx = theta * undistorted_point[0] / w;
      T dy = theta * undistorted_point[1] / w;

      const T sq_r = (dx * dx) + (dy * dy);
      const T radial_distortion = T(1) + distortion[0] * sq_r +
                                  distortion[1] * sq_r * sq_r +
                                  distortion[2] * sq_r * sq_r * sq_r;
      const T tangential_x = T(2) * distortion[3] * dx * dy +
                             distortion[4] * (sq_r + T(2.0) * dx * dx);
      const T tangential_y = T(2) * distortion[4] * dx * dy +
                             distortion[3] * (sq_r + T(2.0) * dy * dy);

      // Normally we'd compute the distorted point d = (radial * undistorted) +
      // tangential, so we invert this and get undistorted = (d - tangential) /
      // radial;
      //
      undistorted_point[0] =
          (distorted_point[0] - tangential_x) / radial_distortion;
      undistorted_point[1] =
          (distorted_point[1] - tangential_y) / radial_distortion;

      // Keep doing this until we've converged.
      if (ceres::abs(undistorted_point[0] - prev_undistorted_point[0]) <
              1e-10 &&
          ceres::abs(undistorted_point[1] - prev_undistorted_point[1]) <
              1e-10) {
        break;
      }
    }
  }
}

template <typename T>
void FisheyeCamera::TransformPixelToCamera(const T* intrinsics,
                                           const T* distortion, const T* pixel,
                                           T* camera) {
  T distorted_point[2];
  distorted_point[1] = (pixel[1] - intrinsics[7]) / intrinsics[4];
  distorted_point[0] =
      (pixel[0] - intrinsics[6] - distorted_point[1] * intrinsics[3]) /
      intrinsics[0];

  FisheyeCamera::UndistortPoint(distortion, distorted_point, camera);
  camera[2] = T(1.0);
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
