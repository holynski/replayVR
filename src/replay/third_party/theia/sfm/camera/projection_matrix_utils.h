// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_SFM_CAMERA_PROJECTION_MATRIX_UTILS_H_
#define THEIA_SFM_CAMERA_PROJECTION_MATRIX_UTILS_H_

#include <Eigen/Core>

#include "replay/third_party/theia/sfm/types.h"

namespace theia {

// Extract intrinsics from a calibration matrix of the following form:
//   [focal_length            skew               principal_point_x]
//   [0            focal_length * aspect_ratio   principal_point_y]
//   [0                         0                              1.0]
void IntrinsicsToCalibrationMatrix(const double focal_length,
                                   const double skew,
                                   const double aspect_ratio,
                                   const double principal_point_x,
                                   const double principal_point_y,
                                   Eigen::Matrix3d* calibration_matrix);

void CalibrationMatrixToIntrinsics(const Eigen::Matrix3d& calibration_matrix,
                                   double* focal_length,
                                   double* skew,
                                   double* aspect_ratio,
                                   double* principal_point_x,
                                   double* principal_point_y);

bool DecomposeProjectionMatrix(const Matrix3x4d pmatrix,
                               Eigen::Matrix3d* calibration_matrix,
                               Eigen::Vector3d* rotation,
                               Eigen::Vector3d* position);

bool ComposeProjectionMatrix(const Eigen::Matrix3d& calibration_matrix,
                             const Eigen::Vector3d& rotation,
                             const Eigen::Vector3d& position,
                             Matrix3x4d* pmatrix);

// Projects a 3x3 matrix to the rotation matrix in SO3 space with the closest
// Frobenius norm. For a matrix with an SVD decomposition M = USV, the nearest
// rotation matrix is R = UV'.
Eigen::Matrix3d ProjectToRotationMatrix(const Eigen::Matrix3d& matrix);
}  // namespace theia

#endif  // THEIA_SFM_CAMERA_PROJECTION_MATRIX_UTILS_H_
