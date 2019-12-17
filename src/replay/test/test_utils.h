#pragma once

#include <glog/logging.h>
#include <Eigen/Dense>
#include "gtest/gtest.h"

namespace replay {
namespace test {

// Assert that values of the two matrices are nearly the same.
template <typename Derived>
void ExpectMatricesNear(const Eigen::MatrixBase<Derived>& a,
                        const Eigen::MatrixBase<Derived>& b, double tolerance) {
  ASSERT_EQ(a.rows(), b.rows());
  ASSERT_EQ(a.cols(), b.cols());
  for (int i = 0; i < a.rows(); i++)
    for (int j = 0; j < a.cols(); j++)
      ASSERT_NEAR(a(i, j), b(i, j), tolerance)
          << "Entry (" << i << ", " << j << ") did not meet the tolerance!";
}

void ExpectArraysNear(int n, const double* a, const double* b,
                      double tolerance) {
  ASSERT_GT(n, 0);
  CHECK(a);
  CHECK(b);
  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(a[i], b[i], tolerance) << "i = " << i;
  }
}

// Expects that for all i = 1,.., n - 1
//
//   |p[i] / max_norm_p - q[i] / max_norm_q| < tolerance
//
// where max_norm_p and max_norm_q are the max norms of the arrays p
// and q respectively.
bool ArraysEqualUpToScale(int n, const double* p, const double* q,
                          double tolerance) {
  Eigen::Map<const Eigen::VectorXd> p_vec(p, n);
  Eigen::Map<const Eigen::VectorXd> q_vec(q, n);

  // Use the cos term in the dot product to determine equality normalized for
  // scale.
  const double cos_diff = p_vec.dot(q_vec) / (p_vec.norm() * q_vec.norm());
  return std::abs(cos_diff) >= 1.0 - tolerance;
}

}  // namespace test
}  // namespace replay
