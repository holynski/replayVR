#include "replay/io/read_capreal.h"
#include "replay/camera/pinhole_camera.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/strings.h"
#include "replay/util/filesystem.h"
#include "replay/util/image_cache.h"

#include <glog/logging.h>
#include <Eigen/Geometry>

namespace replay {

bool ReadCapreal(const std::string& filename,
                 const ImageCache& cache, Reconstruction* recon) {
  std::ifstream reader(filename);
  if (!reader.is_open()) {
    LOG(ERROR) << "Couldn't open file: " << filename;
    return false;
  }
  std::string line;

  int i = 0;
  while (getline(reader, line)) {
    if (line[0] == '#') {
      continue;
    }

    std::vector<std::string> tokens = Tokenize(line);
    CHECK_EQ(tokens.size(), 13);

    PinholeCamera* cam = new PinholeCamera();
    cam->SetName(tokens[0]);

    Eigen::Vector3d position(
        Eigen::Vector3d(stof(tokens[1]), stof(tokens[2]), stof(tokens[3])));

    Eigen::Quaterniond orientation =
        Eigen::AngleAxisd(M_PI * -stof(tokens[4]) / 180.0,
                          Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(M_PI * -stof(tokens[5]) / 180.0,
                          Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd((M_PI * -stof(tokens[6]) / 180.0),
                          Eigen::Vector3d::UnitZ());
    
    Eigen::Matrix3d rotation = orientation.toRotationMatrix().transpose();
    cam->SetRotation(rotation);
    cam->SetPosition(position);

    cam->SetImageSize(cache.GetImageSize(tokens[0]));
    cam->SetPrincipalPoint(Eigen::Vector2d(0.5,0.5));
    LOG(FATAL) << "Something's wrong with the focal length reading!";

    //cam->SetFocalLength(Eigen::Vector2d(hard_focal / 1920.0, hard_focal / 1080));

    recon->AddCamera(cam);
    i++;
  }

  return true;
}

}  // namespace replay
