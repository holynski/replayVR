#pragma once

#include <Eigen/Dense>

namespace replay {

// Decomposes a rotation matrix to yaw, pitch, roll
Eigen::Vector3f DecomposeRotation(const Eigen::Matrix3f& rotation);

// Composes a rotation matrix from yaw, pitch, roll
Eigen::Matrix3f ComposeRotation(const Eigen::Vector3f& decomposed);

}  // namespace replay

