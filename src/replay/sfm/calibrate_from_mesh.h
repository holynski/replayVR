
#include <opencv2/opencv.hpp>

namespace replay {

class Mesh;
class Camera;

// Uses 3D-2D matches (3D points with UV-coords) to estimate camera intrinsics.
//
// Takes as input a Camera object and a point cloud stored in a Mesh object. The
// Mesh may have triangle faces, but they will be ignored. The mesh must have
// per-vertex UVs for all vertices, otherwise the method will fail.
//
// This method takes 3D-to-2D correspondences (where 3D are the 3D points in the
// mesh, and 2D are the UV coordinates of those points) to solve for camera
// intrinsics.
//
// The camera should have the image size set. The type of the camera passed to
// this function (PinholeCamera, FisheyeCamera, etc) will define what parameters
// are optimized for. The better the estimated parameters (the parameters of the
// camera when it is passed in), the more likely the optimization will converge
// on the correct solution.
//
// Under the hood, this function is running bundle adjustment to minimize the
// error between the UV coordinates and the projected 3D points.

// Sample usage:
//
// FisheyeCamera camera;
// camera.SetImageSize(...)
//
// // Set the camera parameters to your best guess of what they will be.
// camera.SetFocalLengthFromFOV(...) // the approx FOV (in degrees)
// camera.SetPrincipalPoint(...) // a good guess is the image center
// camera.SetSkew(...) // a good guess is 0
//
// Mesh mesh;
// mesh.Load("pointcloud.ply");
// CalibrateFromMeshUVs(mesh, &camera);
//
void CalibrateFromMesh(const Mesh& mesh, Camera* camera);

// Visualizes the error for a calibration produced by the function above.
cv::Mat3b VisualizeMeshCalibrationError(const Mesh& mesh, const Camera* camera);

}  // namespace replay
