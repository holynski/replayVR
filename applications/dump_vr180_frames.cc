#include <glog/logging.h>
#include <openvr.h>
#include <replay/util/filesystem.h>
#include <replay/util/strings.h>
#include <replay/vr_180/vr_180_video_reader.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include "replay/rendering/opengl_context.h"
#include "replay/rendering/vr180_undistorter.h"
#include "replay/camera/pinhole_camera.h"

DEFINE_string(video_file, "", "Spherical video file to parse");
DEFINE_string(output_folder, "", "Output folder to save frames");
DEFINE_double(fov, 170, "Field of view of undistorted camera");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_video_file.length() > 0) << "Filename is empty.";
  replay::VR180VideoReader reader;

  std::shared_ptr<replay::OpenGLContext> renderer =
      std::make_shared<replay::OpenGLContext>();
  CHECK(renderer->Initialize());


  replay::PinholeCamera camera;
  camera.SetPosition(Eigen::Vector3d(0,0,0));
  camera.SetOrientationFromLookAtUpVector(Eigen::Vector3d(0,0,-1), Eigen::Vector3d(0,-1,0));
  camera.SetImageSize(Eigen::Vector2i(2160, 2160));
  camera.SetFocalLengthFromFOV(Eigen::Vector2d(FLAGS_fov, FLAGS_fov));
  camera.SetPrincipalPoint(Eigen::Vector2d(1080, 1080));
  LOG(INFO) << camera.GetFocalLength();

  replay::VR180Undistorter undistorter(renderer, camera);
  CHECK(undistorter.Open(FLAGS_video_file));

  cv::Mat3b left_eye, right_eye;
  int filenumber = 0;
  std::string filename;

  while (undistorter.UndistortFrame(&left_eye, &right_eye)) {
    filename = replay::PadZeros(filenumber, 6);
    LOG(INFO) << "Saving: " << replay::JoinPath(FLAGS_output_folder, filename + "_left.png");
    cv::imwrite(replay::JoinPath(FLAGS_output_folder, filename + "_left.png"),
                left_eye);
    LOG(INFO) << "Saving: " << replay::JoinPath(FLAGS_output_folder, filename + "_right.png");
    cv::imwrite(replay::JoinPath(FLAGS_output_folder, filename + "_right.png"),
                right_eye);
    filenumber++;
  }
  return 0;
}
