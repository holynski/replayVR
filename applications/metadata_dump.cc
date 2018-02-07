#include <glog/logging.h>
#include <replay/io/spherical_video_reader.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>

DEFINE_string(video_file, "", "Spherical video file to parse");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_video_file.length() > 0) << "Filename is empty.";
  replay::SphericalVideoReader reader(FLAGS_video_file);
  cv::Mat3b image;
  Eigen::Vector3f angle_axis;
  while (reader.GetOrientedFrame(image, angle_axis)) {
    cv::imshow("image", image);
    LOG(INFO) << angle_axis;
    cv::waitKey();
  }
  return 0;
}
