#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <unordered_set>
#include <vector>
#include "replay/camera/camera.h"
#include "replay/sfm/tracked_point.h"
#include <cereal/types/vector.hpp>

namespace replay {
class Mesh;

class Reconstruction {
 public:
  bool Load(const std::string& filename);
  bool Save(const std::string& filename) const;
  void SetPoints(const std::vector<TrackedPoint*> points);
  void SetCameras(const std::vector<Camera*> cameras);
  void AddCamera(Camera* camera);
  void AddPoint(TrackedPoint* point);

  int NumCameras() const;
  int NumPoints() const;
  bool HasExposure() const;

  void SaveTrajectoryMesh(const std::string& filename) const;
  void SaveMesh(const std::string& filename) const;

  const Camera& GetCamera(const int index) const;
  Camera* GetCameraMutable(const int index);

  // Returns the centroid of camera frusta
  Eigen::Vector3d GetCameraCentroid() const;

  // Returns a list of cameras that
  std::vector<Camera*> FindSimilarViewpoints(
      const Camera* camera, const int angle_threshold = 10) const;

  Mesh CreateFrustumMesh() const;
  Mesh CreatePointCloud() const;

 private:
  std::vector<Camera*> cameras_;
  std::vector<TrackedPoint*> points_;
};
}  // namespace replay
