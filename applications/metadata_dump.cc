#include <glog/logging.h>
#include <replay/vr_180/vr_180_video_reader.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include "replay/rendering/opengl_renderer.h"
#include "replay/rendering/stereo_video_angular_renderer.h"

DEFINE_string(video_file, "", "Spherical video file to parse");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_video_file.length() > 0) << "Filename is empty.";
  replay::VR180VideoReader reader;

  std::shared_ptr<replay::OpenGLRenderer> renderer = std::make_shared<replay::OpenGLRenderer>();
  renderer->Initialize();

  replay::StereoVideoAngularRenderer stereo_renderer(renderer);
  CHECK(stereo_renderer.Initialize(FLAGS_video_file));

  theia::Camera camera;
  camera.SetFocalLength(500);
  camera.SetPrincipalPoint(500, 500);
  camera.SetImageSize(1000, 1000);

  double angle = 0;
  while (true) {
    Eigen::Vector3f lookat(cos(angle), 0, sin(angle));
    stereo_renderer.RenderEye(camera, 0, lookat);
    //stereo_renderer.RenderEye(camera, 1, lookat);

    //int keyvalue = cv::waitKey();
    //if (keyvalue == ' ') {
      //break;
    //} else if (keyvalue == 'z') {
      angle += 0.001;
    //} else if (keyvalue == 'x') {
      //angle -= 0.01;
    //}
  }
  return 0;
}
