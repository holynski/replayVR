#include "replay/rendering/vr180_renderer.h"
#include "replay/rendering/vr_context.h"
#include <Eigen/Dense>
#include <glog/logging.h>
#include <openvr.h>
#include <replay/vr_180/vr_180_video_reader.h>
#include <stdio.h>
#include <stdlib.h>

DEFINE_string(video_file, "", "Spherical video file to parse");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_video_file.length() > 0) << "Filename is empty.";
  replay::VR180VideoReader reader;

  std::shared_ptr<replay::VRContext> renderer =
      std::make_shared<replay::VRContext>();
  CHECK(renderer->Initialize());

  replay::VR180Renderer stereo_renderer(renderer);
  CHECK(stereo_renderer.Initialize(FLAGS_video_file));

  while (true) {
    stereo_renderer.Render();
  }
  return 0;
}
