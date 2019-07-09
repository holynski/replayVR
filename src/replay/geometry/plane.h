#pragma once

#include <Eigen/Core>

namespace replay {

class Mesh;

class Plane {
 public:
  Plane(const float nx, const float ny, const float nz, const float d);
  Plane(const Eigen::Vector3f& point, const Eigen::Vector3f& normal);
  Plane(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2,
        const Eigen::Vector3f& p3);

  Plane(const std::vector<Eigen::Vector3f>& points);

  Eigen::Vector3f Intersect(const Eigen::Vector3f& ray_origin,
                            const Eigen::Vector3f& ray_direction) const;
  float UnsignedDistance(const Eigen::Vector3f& point) const;
  Eigen::Vector3f ProjectToPlane(const Eigen::Vector3f& point) const;

  Eigen::Vector4f GetCoefficients() const;

  Mesh GetMesh(const Eigen::Vector3f& center = Eigen::Vector3f(0, 0, 0),
               const float extent = 5000) const;

 private:
  bool FitToPoints(const std::vector<Eigen::Vector4f>& Points,
                   Eigen::Vector3f& Basis1, Eigen::Vector3f& Basis2,
                   float& NormalEigenvalue, float& ResidualError);
  Eigen::Vector3f normal_;
  float d_;
};

}  // namespace replay
