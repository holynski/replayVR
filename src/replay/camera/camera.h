#pragma once

#include <Eigen/Dense>

namespace replay {

enum CameraType {
  PINHOLE,
};

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

  // Default constructor sets type to PINHOLE.
  Camera();
  Camera(const CameraType type);
  CameraType GetType() const;

  //
  // Functions for getting/setting intrinsics.
  //

  // Returns the X,Y focal lengths
  Eigen::Vector2f GetFocalLength() const;

  // Returns the X,Y principal points, where 0,0 is the lower-left corner
  Eigen::Vector2f GetPrincipalPoint() const;

  // The W,H size of the image
  Eigen::Vector2i GetImageSize() const;

  // Returns a 3x3 matrix with the camera intrinsics (focal lengths and
  // principal points).
  Eigen::Matrix3f GetIntrinsics() const;

  // Returns the field of view in the horizontal and vertical axes
  Eigen::Vector2f GetFOV() const;  // (in degrees)

  // Sets the focal lengths in X,Y
  void SetFocalLength(const Eigen::Vector2f& focal);

  // Sets the focal length from the horizontal and vertical FOVs (in degrees).
  void SetFocalLengthFromFOV(const Eigen::Vector2f& fov);

  // Sets the X,Y principal point. (0,0) is the bottom left corner.
  void SetPrincipalPoint(const Eigen::Vector2f& principal);

  // Sets the 3x3 camera intrinsics. Skew is currently unsupported.
  void SetIntrinsics(const Eigen::Matrix3f& intrinsics);

  // Sets the W,H image size in pixels
  void SetImageSize(const Eigen::Vector2i& image_size);

  //
  // Functions for getting/setting extrinsics
  //

  // Returns a 3x3 rotation matrix (world to camera)
  Eigen::Matrix3f GetRotation() const;

  // Returns the orientation of the camera in quaternion format.
  Eigen::Quaternionf GetOrientation() const;

  // Returns the XYZ center of the camera.
  Eigen::Vector3f GetPosition() const;

  // Returns a 3x4 matrix with rotation and translation (world to camera)
  Eigen::Matrix4f GetExtrinsics() const;

  // Sets the camera orientation from a 3x3 rotation matrix
  void SetRotation(const Eigen::Matrix3f& rotation);

  // Sets the camera orientation from a unit quaternion
  void SetOrientation(const Eigen::Quaternionf& orientation);

  // Sets the camera orientation from a lookat-up vector pair
  void SetOrientationFromLookAtUpVector(const Eigen::Vector3f& lookat,
                                        const Eigen::Vector3f& up);

  // Sets the camera center position in world coordinate space
  void SetPosition(const Eigen::Vector3f& position);

  // Sets the 3x4 camera extrinsics matrix
  void SetExtrinsics(const Eigen::Matrix4f& extrinsics);

  //
  // Functions for OpenGL
  //

  // Returns an OpenGL-style projection matrix (does not include
  // rotation/translation of camera).
  Eigen::Matrix4f GetOpenGlProjection(float near_clip = 0.01f,
                                      float far_clip = 100.0f) const;

  // Returns the camera extrinsics in the OpenGL coordinate system.
  Eigen::Matrix4f GetOpenGlExtrinsics() const;

  // Returns the full MVP matrix used to render this virtual camera in OpenGL.
  // It is the composition of the two functions above.
  Eigen::Matrix4f GetOpenGlMvpMatrix() const;

  // Functions for getting/setting distortion parameters
  // TODO(holynski): Implement storage for distortion parameters, and
  // undistortion, etc.

 private:
  CameraType type_;
  Eigen::Matrix3f intrinsics_;
  Eigen::Matrix4f extrinsics_;
  Eigen::Vector2i image_size_;
};

}  // namespace replay
