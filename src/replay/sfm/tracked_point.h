#pragma once

#include "replay/camera/camera.h"

#include <Eigen/Dense>
#include <unordered_map>

namespace replay {

// A class that represents a 3D point and all its observations across multiple
// images. The class contains:
// - the 3D location of the point in world-space
// - pointers to all the cameras that see it,
// - information about where in each camera/image it was seen
// - the per-camera depth of the point
//
// The decision to store both per-camera depth and 3D position comes from the
// fact that the depth observation of a single camera, whether it comes from a
// stereo pair, active depth sensor, or monocular depth estimation, doesn't
// always agree with the final estimated 3D position of the point when
// bundle-adjusted -- in the same way the 3D point doesn't always lie along the
// ray defined by the pixel coordinates of any single camera.
class TrackedPoint {
 public:
  // Returns the 3D point in world-space
  const Eigen::Vector3d& GetPoint() const;

  // Sets the location of the 3D point
  void SetPoint(const Eigen::Vector3d& point);

  // Returns the number of cameras that see this point.
  int NumObservations() const;

  // Returns the 2D observation of a particular point (X,Y). The third
  // coordinate (Z) is used to store depth when triangulating. Careful, this
  // isn't projective depth (Z-depth), but rather "length-along-the-ray" depth.
  const Eigen::Vector3d& GetObservation(const Camera* camera) const;
  const Eigen::Vector2d GetObservation2D(const Camera* camera) const;

  // Returns a map of all the (Camera - 2D point) observation pairs
  const std::unordered_map<const Camera*, Eigen::Vector3d>& GetObservations()
      const;

  // Returns true is a particular camera sees this point
  bool HasObservation(const Camera* camera) const;

  // Adds/sets a camera to see this tracked point at the 2D coordiante defined
  // by 'point'. The observation depth will be set to 0. This will not affect
  // the 3D position of the TrackedPoint as accessed by GetPoint().
  void SetObservation(const Camera* camera, const Eigen::Vector2d& point);
  void SetObservation(const Camera* camera, const cv::Point2f& point);

  // Adds/sets a camera to see this tracked point at the 2D coordiante defined
  // by 'point'. The depth of the point will be set to the Z coordinate of
  // 'point'. Setting the observation depth will not affect the 3D position of
  // the TrackedPoint as accessed by GetPoint().
  void SetObservation(const Camera* camera, const Eigen::Vector3d& point);

  // A function for triangulating a point seen by multiple cameras.
  //
  // If the "cameras" parameter is provided, this will indicate that the point
  // should be triangulated for a particular subset of the cameras that observe
  // it. If it is not provided, all the observing cameras will be used.
  //
  // This function will overwrite the per-camera observed depth (the Z
  // coordinate of each observation). It will be overwritten to the ray-distance
  // of the camera to the triangulated 3D point.
  Eigen::Vector3d Triangulate(
      const std::vector<const Camera*> cameras = std::vector<const Camera*>());

 private:
  std::unordered_map<const Camera*, Eigen::Vector3d> observations_;
  Eigen::Vector3d point_;
};
}  // namespace replay
