#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace replay {

enum CameraType { PINHOLE, FISHEYE };

// A class to hold camera intrinsics / extrinsics.
// All coordinate systems are Z-forward, Y-up, X-right.
//
// To initialize a camera:
//  - Initialize the extrinsics (see extrinsics instructions below)
//  - Initialize the intrinsics (see intrinsics instructions below)
//
// To fully initialize the intrinsics, you must:
// (Either)
//    SetFocalLength / SetFocalLengthFromFOV
//    SetPrincipalPoint
//    SetImageSize
// (Or)
//    SetIntrinsics
//    SetImageSize
//
// To fully initialize the extrinsics, you must:
// (Either)
//    SetRotation / SetOrientation/ SetOrientationFromLookat
//    SetPosition
// (Or)
//    SetExtrinsics
//
class Camera {
 public:
  Camera();
  // Returns the camera type
  CameraType GetType() const;

  //
  // Functions for getting/setting intrinsics.
  //

  // Returns the X,Y focal lengths
  Eigen::Vector2d GetFocalLength() const;

  // Returns the X,Y principal points, where 0,0 is the lower-left corner
  Eigen::Vector2d GetPrincipalPoint() const;

  // The W,H size of the image
  Eigen::Vector2i GetImageSize() const;

  // Returns a 3x3 matrix with the camera intrinsics (focal lengths and
  // principal points).
  const Eigen::Matrix3d& GetIntrinsicsMatrix() const;

  // Returns a 3x3 matrix with the camera intrinsics (focal lengths and
  // principal points).
  double GetSkew() const;

  // Returns the field of view in the horizontal and vertical axes
  virtual Eigen::Vector2d GetFOV() const = 0;  // (in degrees)

  // Gets the distortion coefficients
  const std::vector<double>& GetDistortionCoeffs() const;

  // Sets the focal lengths in X,Y
  void SetFocalLength(const Eigen::Vector2d& focal);

  // Sets the focal length from the horizontal and vertical FOVs (in degrees).
  virtual void SetFocalLengthFromFOV(const Eigen::Vector2d& fov) = 0;

  // Sets the X,Y principal point. (0,0) is the bottom left corner.
  void SetPrincipalPoint(const Eigen::Vector2d& principal);

  // Sets the skew in the projection matrix
  void SetSkew(const double skew);

  // Sets the 3x3 camera intrinsics. Skew is currently unsupported.
  void SetIntrinsicsMatrix(const Eigen::Matrix3d& intrinsics);

  // Sets the W,H image size in pixels
  void SetImageSize(const Eigen::Vector2i& image_size);

  // Sets the distortion coefficients
  virtual void SetDistortionCoeffs(const std::vector<double>& coeffs) = 0;

  //
  // Functions for getting/setting extrinsics
  //

  // Returns a 3x3 rotation matrix (world to camera)
  Eigen::Matrix3d GetRotation() const;

  // Returns the orientation of the camera in quaternion format.
  Eigen::Quaterniond GetOrientation() const;

  // Returns the XYZ center of the camera.
  Eigen::Vector3d GetPosition() const;

  // Returns a 3x4 matrix with rotation and translation (world to camera)
  Eigen::Matrix4d GetExtrinsics() const;

  // Sets the camera orientation from a 3x3 rotation matrix
  void SetRotation(const Eigen::Matrix3d& rotation);

  // Sets the camera orientation from a unit quaternion
  void SetOrientation(const Eigen::Quaterniond& orientation);

  // Sets the camera orientation from a lookat-up vector pair
  void SetOrientationFromLookAtUpVector(const Eigen::Vector3d& lookat,
                                        const Eigen::Vector3d& up);

  // Sets the camera center position in world coordinate space
  void SetPosition(const Eigen::Vector3d& position);

  // Sets the 3x4 camera extrinsics matrix
  void SetExtrinsics(const Eigen::Matrix4d& extrinsics);

  //
  // Functions for OpenGL
  //

  // Returns an OpenGL-style projection matrix (does not include
  // rotation/translation of camera).
  Eigen::Matrix4f GetOpenGlProjection(double near_clip = 0.01f,
                                      double far_clip = 100.0f) const;

  // Returns the camera extrinsics in the OpenGL coordinate system.
  Eigen::Matrix4f GetOpenGlExtrinsics() const;

  // Returns the full MVP matrix used to render this virtual camera in OpenGL.
  // It is the composition of the two functions above.
  Eigen::Matrix4f GetOpenGlMvpMatrix() const;

  //
  // Other useful operations
  //

  // Projects a 3D point into 2D coordinates
  virtual Eigen::Vector2d ProjectPoint(
      const Eigen::Vector3d& point3d) const = 0;

  // Uses the distortion coefficients to undistort an image.
  virtual cv::Mat UndistortImage(const cv::Mat& image) const = 0;

  //
  // Templated functions that can be used for bundle adjustment with Ceres
  //

  // Uses the camera extrinsics to convert a point from world coordinates to
  // camera coordinates
  template <typename T>
  static void TransformWorldToCamera(const T* extrinsics, const T* world,
                                     T* camera);

  //
  // Accessors for mutable arrays. Useful for speed and in-place modification.
  //

  const double* intrinsics();
  double* mutable_intrinsics();
  const double* distortion_coeffs();
  double* mutable_distortion_coeffs();

 protected:
  CameraType type_;
  Eigen::Matrix3d intrinsics_;
  Eigen::Matrix4d extrinsics_;
  Eigen::Vector2i image_size_;
  std::vector<double> distortion_coeffs_;
};

// Assumes column-major storage order of the extrinsics matrix
template <typename T>
void Camera::TransformWorldToCamera(const T* extrinsics, const T* world,
                                    T* camera) {
  camera[0] = world[0] * extrinsics[0] + world[1] * extrinsics[4] +
              world[2] * extrinsics[8] + world[3] * extrinsics[12];
  camera[1] = world[0] * extrinsics[1] + world[1] * extrinsics[5] +
              world[2] * extrinsics[9] + world[3] * extrinsics[13];
  camera[2] = world[0] * extrinsics[2] + world[1] * extrinsics[6] +
              world[2] * extrinsics[10] + world[3] * extrinsics[14];
  camera[3] = world[0] * extrinsics[3] + world[1] * extrinsics[7] +
              world[2] * extrinsics[11] + world[3] * extrinsics[15];
}

}  // namespace replay
