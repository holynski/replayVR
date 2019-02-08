#include "replay/io/read_bundler.h"
#include "replay/camera/pinhole_camera.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/filesystem.h"
#include "replay/util/image_cache.h"
#include "replay/util/strings.h"

#include <glog/logging.h>
#include <Eigen/Geometry>

namespace replay {

namespace {

std::vector<std::string> GetFilenamesFromImageList(
    const std::string& image_list) {
  std::vector<std::string> filenames;
  std::ifstream reader(image_list);
  if (!reader.is_open()) {
    LOG(FATAL) << "Couldn't open file: " << image_list;
  }

  std::string line;
  while (!(line = GetLine(&reader)).empty()) {
    filenames.push_back(GetFilenameFromPath(line));
  }
  reader.close();
  return filenames;
}

}  // namespace

bool ReadBundler(const std::string& bundler_filename,
                 const std::string& image_list, const ImageCache& cache,
                 Reconstruction* recon) {
  std::ifstream reader(bundler_filename);
  if (!reader.is_open()) {
    LOG(ERROR) << "Couldn't open file: " << bundler_filename;
    return false;
  }

  const std::vector<std::string> image_filenames =
      GetFilenamesFromImageList(image_list);

  std::string line;

  // Read header
  line = GetLine(&reader);
  CHECK_EQ(line, "# Bundle file v0.3");

  // Read <numcameras> <numpoints>
  line = GetLine(&reader);
  std::vector<std::string> tokens = Tokenize(line, ' ');
  CHECK_EQ(tokens.size(), 2);
  const int num_cameras = stoi(tokens[0]);
  const int num_points = stoi(tokens[1]);

  const int desired_cameras = std::min(num_cameras, 200);
  const int to_be_skipped = num_cameras - desired_cameras;
  int skip_val = 1;
  if (desired_cameras < num_cameras) {
    skip_val = std::ceil(static_cast<float>(num_cameras) / desired_cameras);
  }

  // Read the cameras
  for (int i = 0; i < num_cameras; i++) {
    line = GetLine(&reader);
    tokens = Tokenize(line, ' ');
    PinholeCamera* cam = new PinholeCamera();
    cam->SetFocalLength(Eigen::Vector2d(stof(tokens[0]), stof(tokens[0])));
    std::vector<double> distortion;
    distortion.push_back(stof(tokens[1]));
    distortion.push_back(stof(tokens[2]));
    cam->SetDistortionCoeffs(distortion);
    Eigen::Matrix3d rotation;
    for (int row = 0; row < 3; row++) {
      line = GetLine(&reader);
      tokens = Tokenize(line, ' ');
      for (int col = 0; col < 3; col++) {
        rotation(row, col) = stof(tokens[col]);
      }
    }

    Eigen::Quaterniond quat(rotation.cast<double>());
    std::swap(quat.x(), quat.w());
    std::swap(quat.y(), quat.z());
    quat.x() *= -1;
    quat.w() *= -1;
    quat.z() *= -1;
    cam->SetOrientation(quat);

    // TODO(holynski): Fix this hacky transform
    line = GetLine(&reader);
    tokens = Tokenize(line, ' ');
    Eigen::Vector3d translation;
    for (int col = 0; col < 3; col++) {
      translation(col) = stof(tokens[col]);
    }
    translation = -rotation.transpose() * translation;
    // cam->SetRotation(rotation);
    cam->SetPosition(translation);
    cam->SetImageSize(cache.GetImageSize(image_filenames[i]));
    cam->SetName(image_filenames[i]);
    cam->SetPrincipalPoint(Eigen::Vector2d(0.5, 0.5));
    if (i < to_be_skipped / 2) {
      continue;
    }
    if (i > to_be_skipped / 2 + desired_cameras) {
      break;
    }
    recon->AddCamera(cam);
  }

  // Read the points
  for (int i = 0; i < num_points; i++) {
  }
  reader.close();
  return true;
}

}  // namespace replay
