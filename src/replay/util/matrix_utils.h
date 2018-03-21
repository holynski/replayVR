#pragma once

#include <Eigen/Dense>

namespace replay {

// Decomposes a rotation matrix to yaw, pitch, roll
Eigen::Vector3f DecomposeRotation(const Eigen::Matrix3f& rotation) {
  Eigen::Vector3f decomposed;
  decomposed[0] = atan2(rotation(2, 1), rotation(2, 2));
  decomposed[1] = atan2(-rotation(2, 0),
                        Eigen::Vector2f(rotation(2, 1), rotation(2, 2)).norm());
  decomposed[2] = atan2(rotation(1, 0), rotation(0, 0));
  return decomposed;
}

// Composes a rotation matrix from yaw, pitch, roll
Eigen::Matrix3f ComposeRotation(const Eigen::Vector3f& decomposed) {
  Eigen::Matrix3f yaw, pitch, roll;
  yaw << 1, 0, 0, 0, cos(decomposed[0]), -sin(decomposed[0]), 0,
      sin(decomposed[0]), cos(decomposed[0]);
  pitch << cos(decomposed[1]), 0, sin(decomposed[1]), 0, 1, 0,
      -sin(decomposed[1]), 0, cos(decomposed[1]);
  roll << cos(decomposed[2]), -sin(decomposed[2]), 0, sin(decomposed[2]),
      cos(decomposed[2]), 0, 0, 0, 1;
  return yaw * pitch * roll;
}

}  // namespace replay
