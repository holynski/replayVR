#include "replay/sfm/tracked_point.h"

#include "replay/camera/fisheye_camera.h"
#include "replay/camera/pinhole_camera.h"

#include <glog/logging.h>
#include <Eigen/Dense>
#include <unordered_map>

namespace replay {

const Eigen::Vector3d& TrackedPoint::GetPoint() const { return point_; }

const std::unordered_map<const Camera*, Eigen::Vector3d>&
TrackedPoint::GetObservations() const {
  return observations_;
}

int TrackedPoint::NumObservations() const { return observations_.size(); }

const Eigen::Vector3d& TrackedPoint::GetObservation(
    const replay::Camera* camera) const {
  CHECK(observations_.find(camera) != observations_.end())
      << "Camera not found! Check if the observation exists first using "
         "HasObservation().";
  return observations_.at(camera);
}
const Eigen::Vector2d TrackedPoint::GetObservation2D(
    const replay::Camera* camera) const {
  CHECK(observations_.find(camera) != observations_.end())
      << "Camera not found! Check if the observation exists first using "
         "HasObservation().";
  return observations_.at(camera).head(2);
}
bool TrackedPoint::HasObservation(const replay::Camera* camera) const {
  return (observations_.find(camera) != observations_.end());
}
void TrackedPoint::SetObservation(const replay::Camera* camera,
                                  const Eigen::Vector3d& point) {
  observations_[camera] = point;
}
void TrackedPoint::SetObservation(const replay::Camera* camera,
                                  const Eigen::Vector2d& point) {
  observations_[camera] = Eigen::Vector3d(point.x(), point.y(), 0);
}

void TrackedPoint::SetObservation(const replay::Camera* camera,
                                  const cv::Point2f& point) {
  observations_[camera] = Eigen::Vector3d(point.x, point.y, 0);
}

void TrackedPoint::SetPoint(const Eigen::Vector3d& point) { point_ = point; }

Eigen::Vector3d TrackedPoint::Triangulate(
    const std::vector<const Camera*> cameras) {
  // We can't triangulate with only one camera.
  CHECK_NE(cameras.size(), 1);

  Eigen::Matrix4d A;
  A.setZero();
  Eigen::Vector4d b;
  b.setZero();
  for (const auto& observation : observations_) {
    // If the cameras have been specified, check to see if the camera we're
    // looking at is in the list. If not, skip it.
    if (cameras.size() != 0 && std::find(cameras.begin(), cameras.end(),
                                         observation.first) == cameras.end()) {
      continue;
    }
    Eigen::Vector3d ray_direction =
        observation.first->PixelToWorldRay(observation.second.head<2>());
    const Eigen::Vector4d ray_direction_homog(
        ray_direction.x(), ray_direction.y(), ray_direction.z(), 0);
    const Eigen::Matrix4d A_term =
        Eigen::Matrix4d::Identity() -
        ray_direction_homog * ray_direction_homog.transpose();
    A += A_term;
    b += A_term * observation.first->GetPosition().homogeneous();
  }
  Eigen::LLT<Eigen::Matrix4d> linear_solver(A);
  if (linear_solver.info() != Eigen::Success) {
    LOG(ERROR) << "Failed triangulation!";
    point_ = Eigen::Vector3d(0, 0, 0);
    return point_;
  }

  point_ = linear_solver.solve(b).hnormalized();

  if (linear_solver.info() != Eigen::Success) {
    LOG(ERROR) << "Failed triangulation!";
    point_ = Eigen::Vector3d(0, 0, 0);
    return point_;
  }

  // Set the per-camera observations to have the appropriate depth
  for (auto& observation : observations_) {
    if (cameras.size() != 0 && std::find(cameras.begin(), cameras.end(),
                                         observation.first) == cameras.end()) {
      continue;
    }
    observation.second.z() =
        (point_ - observation.first->GetPosition())
            .dot(observation.first->GetLookAt().normalized());
    if (observation.second.z() <= 0) {
      point_ = Eigen::Vector3d(0,0,0);
      return point_;
    }
  }
  return point_;
}
}  // namespace replay
