#include "replay/sfm/calibrate_from_mesh.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "replay/camera/fisheye_camera.h"
#include "replay/camera/pinhole_camera.h"
#include "replay/mesh/mesh.h"

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
  bool operator()(const T* const intrinsics, const T* const distortion,
                  T* residuals) const {
    T predicted[2];
    T point3d[3];
    point3d[0] = T(point3d_[0]);
    point3d[1] = T(point3d_[1]);
    point3d[2] = T(point3d_[2]);
    switch (camera_type_) {
      case CameraType::FISHEYE:
        FisheyeCamera::TransformCameraToPixel(intrinsics, distortion, point3d,
                                              predicted);
        break;
      case CameraType::PINHOLE:
        PinholeCamera::TransformCameraToPixel(intrinsics, distortion, point3d,
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

class ViewReprojectionError : public ceres::IterationCallback {
 public:
  ViewReprojectionError(const double* intrinsics, const double* distortion,
                        const float* points2d, const float* points3d,
                        const int num_pts, const CameraType type,
                        const int image_width, const int image_height)
      : camera_type_(type),
        image_width_(image_width),
        image_height_(image_height),
        num_pts_(num_pts),
        intrinsics_(intrinsics),
        distortion_(distortion),
        points2d_(points2d),
        points3d_(points3d) {}

  ~ViewReprojectionError() {}
  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    cv::Mat3b img(image_height_, image_width_, cv::Vec3b(0, 0, 0));
    for (int i = 0; i < num_pts_; i++) {
      double projected[2];
      double point3d[3];

      point3d[0] = points3d_[3 * i + 0];
      point3d[1] = -points3d_[3 * i + 1];
      point3d[2] = -points3d_[3 * i + 2];

      switch (camera_type_) {
        case CameraType::FISHEYE:
          FisheyeCamera::TransformCameraToPixel(intrinsics_, distortion_,
                                                point3d, projected);
          break;
        case CameraType::PINHOLE:
          PinholeCamera::TransformCameraToPixel(intrinsics_, distortion_,
                                                point3d, projected);
          break;
      }

      cv::circle(img,
                 cv::Point(points2d_[2 * i] * image_width_,
                           (1.0 - points2d_[2 * i + 1]) * image_height_),
                 2, cv::Scalar(0, 0, 255), -1);
      cv::line(img, cv::Point(projected[0], projected[1]),
               cv::Point(points2d_[2 * i] * image_width_,
                         (1.0 - points2d_[2 * i + 1]) * image_height_),
               cv::Scalar(0, 0, 255), 2);
      cv::circle(img,
                 cv::Point(points2d_[2 * i] * image_width_,
                           (1.0 - points2d_[2 * i + 1]) * image_height_),
                 2, cv::Scalar(0, 255, 0), -1);
    }
    cv::resize(img, img, cv::Size(), 0.7, 0.7);
    cv::imshow("img", img);
    cv::imwrite("/Users/holynski/error.png", img);
    cv::waitKey(10);
    return ceres::SOLVER_CONTINUE;
  }

 private:
  CameraType camera_type_;
  const int image_width_;
  const int image_height_;
  const int num_pts_;
  const double* intrinsics_;
  const double* distortion_;
  const float* points2d_;
  const float* points3d_;
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
  double* distortion = camera->mutable_distortion_coeffs();
  const int num_distortion_coeffs = camera->GetDistortionCoeffs().size();

  ViewReprojectionError callback(intrinsics, distortion, points2d, points3d,
                                 num_points, camera->GetType(), image_size.x(),
                                 image_size.y());

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (int i = 0; i < num_points; ++i) {
    ceres::CostFunction* cost_function;
    switch (camera->GetType()) {
      case CameraType::PINHOLE:
        cost_function =
            (new ceres::AutoDiffCostFunction<
                MeshProjectionError, 2, 9, PinholeCamera::kNumDistortionCoeffs>(
                new MeshProjectionError(
                    points2d[2 * i + 0] * image_size[0],
                    (1.0f - points2d[2 * i + 1]) * image_size[1],
                    points3d[3 * i + 0], -points3d[3 * i + 1],
                    -points3d[3 * i + 2], camera->GetType())));
        break;
      case CameraType::FISHEYE:
        cost_function =
            (new ceres::AutoDiffCostFunction<
                MeshProjectionError, 2, 9, FisheyeCamera::kNumDistortionCoeffs>(
                new MeshProjectionError(
                    points2d[2 * i + 0] * image_size[0],
                    (1.0f - points2d[2 * i + 1]) * image_size[1],
                    points3d[3 * i + 0], -points3d[3 * i + 1],
                    -points3d[3 * i + 2], camera->GetType())));
        break;
    }
    problem.AddParameterBlock(intrinsics, 9);
    problem.AddParameterBlock(distortion, num_distortion_coeffs);
    problem.SetParameterLowerBound(intrinsics, 0, 1e-6);
    problem.SetParameterLowerBound(intrinsics, 4, 1e-6);
    problem.AddResidualBlock(cost_function, NULL /* squared loss */, intrinsics,
                             distortion);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.update_state_every_iteration = true;
  options.update_state_every_iteration = true;
  options.callbacks.push_back(&callback);
  options.function_tolerance = 1e-8;

  LOG(INFO) << "Intrinsics initialized to: ";
  LOG(INFO) << camera->GetIntrinsicsMatrix();
  LOG(INFO) << "Distortion coefficients initialized to: ";
  for (int i = 0; i < num_distortion_coeffs; i++) {
    LOG(INFO) << distortion[i];
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  LOG(INFO) << "Intrinsics estimated as: ";
  LOG(INFO) << camera->GetIntrinsicsMatrix();
  LOG(INFO) << "Distortion coefficients estimated as: ";
  for (int i = 0; i < num_distortion_coeffs; i++) {
    LOG(INFO) << distortion[i];
  }

  cv::waitKey();
}
}  // namespace replay
