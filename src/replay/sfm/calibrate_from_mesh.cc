#include "replay/sfm/calibrate_from_mesh.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "replay/camera/fisheye_camera.h"
#include "replay/camera/pinhole_camera.h"
#include "replay/geometry/mesh.h"

namespace replay {

namespace {

struct MeshProjectionError {
  MeshProjectionError(double point2d_x, double point2d_y, double point3d_x,
                      double point3d_y, double point3d_z, const CameraType type)
      : camera_type_(type) {
    point2d_[0] = point2d_x;
    point2d_[1] = point2d_y;
    point3d_[0] = point3d_x;
    point3d_[1] = point3d_y;
    point3d_[2] = point3d_z;
  }

  template <typename T>
  bool operator()(const T* extrinsics, const T* const intrinsics,
                  const T* const distortion, T* residuals) const {
    T predicted[3];
    T point3d[3];
    point3d[0] = T(point3d_[0]);
    point3d[1] = T(point3d_[1]);
    point3d[2] = T(point3d_[2]);
    switch (camera_type_) {
      case CameraType::FISHEYE:
        FisheyeCamera::ProjectPoint(extrinsics, intrinsics, distortion, point3d,
                                    predicted);
        break;
      case CameraType::PINHOLE:
        PinholeCamera::ProjectPoint(extrinsics, intrinsics, distortion, point3d,
                                    predicted);
        break;
    }
    // The error is the difference between the predicted and observed position.
    residuals[0] = (predicted[0] - T(point2d_[0]));
    residuals[1] = (predicted[1] - T(point2d_[1]));

    return true;
  }

  const CameraType camera_type_;
  double point2d_[2];
  double point3d_[3];
};

}  // namespace

void CalibrateFromMesh(const Mesh& mesh, Camera* camera) {
  CHECK_NOTNULL(camera);
  CHECK_GT(camera->GetImageSize()[0], 0);
  CHECK_GT(camera->GetImageSize()[1], 0);

  const float* points2d = mesh.uvs();
  const float* points3d = mesh.vertex_positions();
  const int num_points = mesh.NumVertices();
  const Eigen::Vector2i& image_size = camera->GetImageSize();

  double* intrinsics = camera->mutable_intrinsics();
  double* extrinsics = camera->mutable_extrinsics();
  double* distortion = camera->mutable_distortion_coeffs();
  const int num_distortion_coeffs = camera->GetDistortionCoeffs().size();

  ceres::Problem problem;
  for (int i = 0; i < num_points; ++i) {
    ceres::CostFunction* cost_function;
    switch (camera->GetType()) {
      case CameraType::PINHOLE:
        cost_function =
            (new ceres::AutoDiffCostFunction<
                MeshProjectionError, 2, 16, 9,
                PinholeCamera::kNumDistortionCoeffs>(new MeshProjectionError(
                points2d[2 * i + 0] * image_size[0],
                (1.0f - points2d[2 * i + 1]) * image_size[1],
                points3d[3 * i + 0], -points3d[3 * i + 1], -points3d[3 * i + 2],
                camera->GetType())));
        break;
      case CameraType::FISHEYE:
        cost_function =
            (new ceres::AutoDiffCostFunction<
                MeshProjectionError, 2, 16, 9,
                FisheyeCamera::kNumDistortionCoeffs>(new MeshProjectionError(
                points2d[2 * i + 0] * image_size[0],
                (1.0f - points2d[2 * i + 1]) * image_size[1],
                points3d[3 * i + 0], -points3d[3 * i + 1], -points3d[3 * i + 2],
                camera->GetType())));
        break;
    }
    problem.AddParameterBlock(extrinsics, 16);
    problem.AddParameterBlock(intrinsics, 9);
    problem.AddParameterBlock(distortion, num_distortion_coeffs);
    problem.SetParameterLowerBound(intrinsics, 0, 1e-6);
    problem.SetParameterLowerBound(intrinsics, 4, 1e-6);
    // Set the translation constant...only optimize for rotation
    problem.AddResidualBlock(cost_function, NULL /* squared loss */, extrinsics,
                             intrinsics, distortion);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.function_tolerance = 1e-8;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // CHECK_LT(sqrt(summary.final_cost / num_points), 5)
  //<< "Projection error is too high!";
  LOG(INFO) << "Calibrated camera. Final projection error is "
            << sqrt(summary.final_cost / num_points) << " pixels.";
}

cv::Mat3b VisualizeMeshCalibrationError(const Mesh& mesh,
                                        const Camera* camera) {
  const float* points2d = mesh.uvs();
  const float* points3d = mesh.vertex_positions();
  const int num_points = mesh.NumVertices();
  const double* intrinsics = camera->intrinsics();
  const double* extrinsics = camera->extrinsics();
  const double* distortion = camera->distortion_coeffs();
  const Eigen::Vector2i& image_size = camera->GetImageSize();
  const int& image_width = image_size[0];
  const int& image_height = image_size[1];
  cv::Mat3b img(image_height, image_width, cv::Vec3b(0, 0, 0));
  for (int i = 0; i < num_points; i++) {
    double projected[2];
    double point3d[3];

    point3d[0] = points3d[3 * i + 0];
    point3d[1] = -points3d[3 * i + 1];
    point3d[2] = -points3d[3 * i + 2];

    switch (camera->GetType()) {
      case CameraType::FISHEYE:
        FisheyeCamera::ProjectPoint(extrinsics, intrinsics, distortion, point3d,
                                    projected);
        break;
      case CameraType::PINHOLE:
        PinholeCamera::ProjectPoint(extrinsics, intrinsics, distortion, point3d,
                                    projected);
        break;
    }

    cv::circle(img,
               cv::Point(points2d[2 * i] * image_width,
                         (1.0 - points2d[2 * i + 1]) * image_height),
               2, cv::Scalar(0, 0, 255), -1);
    cv::line(img, cv::Point(projected[0], projected[1]),
             cv::Point(points2d[2 * i] * image_width,
                       (1.0 - points2d[2 * i + 1]) * image_height),
             cv::Scalar(0, 0, 255), 2);
    cv::circle(img,
               cv::Point(points2d[2 * i] * image_width,
                         (1.0 - points2d[2 * i + 1]) * image_height),
               2, cv::Scalar(0, 255, 0), -1);
  }
  return img;
}
}  // namespace replay
