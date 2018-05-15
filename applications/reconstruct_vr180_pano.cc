#include <glog/logging.h>
#include <replay/util/filesystem.h>
#include <replay/util/strings.h>
#include <replay/vr_180/vr_180_video_reader.h>
#include <replay/sfm/calibrate_from_mesh.h>
#include <Eigen/Dense>
#include "replay/camera/fisheye_camera.h"
#include "replay/mesh/mesh.h"

DEFINE_string(video_file, "", "Spherical video file to parse");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_video_file.length() > 0) << "Filename is empty.";
  replay::VR180VideoReader reader;
  reader.Open(FLAGS_video_file);

  std::vector<replay::Mesh> meshes = reader.GetMeshes();
  replay::FisheyeCamera camera;

  const int width = reader.GetWidth() / 2;
  const int height = reader.GetHeight();
  camera.SetImageSize(Eigen::Vector2i(width, height));
  camera.SetFocalLengthFromFOV(Eigen::Vector2d(180, 180));
  camera.SetPrincipalPoint(Eigen::Vector2d(width/2, height/2));
  camera.SetSkew(0);

  replay::CalibrateFromMesh(meshes[0], &camera);
  return 1;
}
