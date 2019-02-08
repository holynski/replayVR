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

  virtual Camera* Clone() const = 0;

  // Returns the camera type
  CameraType GetType() const;

  // The filename for the camera, if applicable
  const std::string& GetName() const;
  void SetName(const std::string& name);

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

  // Gets the exposure multiplier
  const Eigen::Vector3f GetExposure() const;

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

  // Gets the exposure multiplier
  void SetExposure(const Eigen::Vector3f& exposure);

  //
  // Functions for getting/setting extrinsics
  //

  // Returns a 3x3 rotation matrix (world to camera)
  Eigen::Matrix3d GetRotation() const;

  // Returns the orientation of the camera in quaternion format.
  Eigen::Quaterniond GetOrientation() const;

  // Returns the translational component of the extrinsics.
  // ** NOT TO BE CONFUSED WITH THE CAMERA POSITION **
  // If you want the position of the camera, see below for GetPosition().
  Eigen::Vector3d GetTranslation() const;

  // Returns the lookat vector for the camera
  Eigen::Vector3d GetLookAt() const;

  // Returns the up vector for the camera
  Eigen::Vector3d GetUpVector() const;

  // Returns the vector pointing to the right (positive X).
  Eigen::Vector3d GetRightVector() const;

  // Returns the XYZ center of the camera.
  Eigen::Vector3d GetPosition() const;

  // Returns a 3x4 matrix with rotation and translation (world to camera)
  Eigen::Matrix4d GetExtrinsics() const;

  // Sets the camera orientation from a 3x3 rotation matrix
  void SetRotation(const Eigen::Matrix3d& rotation);

  // Sets the translational component of the extrinsics matrix
  void SetTranslation(const Eigen::Vector3d& translation);

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
                                      double far_clip = 1000.0f) const;

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

  // Returns the 3D ray a particular pixel in world-space
  virtual Eigen::Vector3d PixelToWorldRay(
      const Eigen::Vector2d& point2d) const = 0;

  // Returns a 3D point in world-space given a pixel and depth. Assumed depth is
  // "ray depth", not projective depth. That is, the distance along the ray
  // between the camera and the point.
  Eigen::Vector3d UnprojectPoint(const Eigen::Vector2d& point2d,
                                 const double depth) const;

  // Returns the angle (in degrees) that a point is from the optical axis. In
  // other words, "what FOV would I need to see this point?".
  double AngleFromOpticalAxis(const Eigen::Vector3d& point) const;

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
  template <typename T>
  static void TransformCameraToWorld(const T* extrinsics, const T* world,
                                     T* camera);

  //
  // Accessors for mutable arrays. Useful for speed and in-place modification.
  //

  const double* extrinsics() const;
  double* mutable_extrinsics();
  const double* intrinsics() const;
  double* mutable_intrinsics();
  const double* distortion_coeffs() const;
  double* mutable_distortion_coeffs();

  static std::string TypeToString(const CameraType type);

 protected:
  CameraType type_;
  Eigen::Matrix3d intrinsics_;
  Eigen::Matrix4d extrinsics_;
  Eigen::Vector2i image_size_;
  std::vector<double> distortion_coeffs_;
  std::string name_;
  Eigen::Vector3f exposure_;
};

// Assumes column-major storage order of the extrinsics matrix
template <typename T>
void Camera::TransformWorldToCamera(const T* extrinsics, const T* world,
                                    T* camera) {
  camera[0] = world[0] * extrinsics[0] + world[1] * extrinsics[4] +
              world[2] * extrinsics[8] + T(1) * extrinsics[12];
  camera[1] = world[0] * extrinsics[1] + world[1] * extrinsics[5] +
              world[2] * extrinsics[9] + T(1) * extrinsics[13];
  camera[2] = world[0] * extrinsics[2] + world[1] * extrinsics[6] +
              world[2] * extrinsics[10] + T(1) * extrinsics[14];
}

template <typename T>
void Camera::TransformCameraToWorld(const T* extrinsics, const T* camera,
                                    T* world) {
  camera[0] = world[0] * extrinsics[0] + world[1] * extrinsics[1] +
              world[2] * extrinsics[2] + T(1) * extrinsics[3];
  camera[1] = world[0] * extrinsics[4] + world[1] * extrinsics[5] +
              world[2] * extrinsics[6] + T(1) * extrinsics[7];
  camera[2] = world[0] * extrinsics[8] + world[1] * extrinsics[9] +
              world[2] * extrinsics[10] + T(1) * extrinsics[11];
}

}  // namespace replay
