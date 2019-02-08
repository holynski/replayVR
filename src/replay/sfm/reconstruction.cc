#include "replay/sfm/reconstruction.h"

#include "replay/camera/camera.h"
#include "replay/camera/fisheye_camera.h"
#include "replay/camera/pinhole_camera.h"
#include "replay/mesh/mesh.h"
#include "replay/sfm/tracked_point.h"

#include <glog/logging.h>
#include <cereal/archives/portable_binary.hpp>
#include <vector>

namespace replay {

bool Reconstruction::Load(const std::string& filename) {
  std::ifstream reader(filename, std::ios::in | std::ios::binary);
  if (!reader.is_open()) {
    LOG(ERROR) << "Couldn't open: " << filename;
    return false;
  }

  int num_cameras;
  reader.read(reinterpret_cast<char*>(&num_cameras), sizeof(int));
  cameras_.resize(num_cameras);
  for (int i = 0; i < num_cameras; i++) {
    std::string camera_type = "UNKNOWN";
    reader.read(&(camera_type[0]), 7);
    if (camera_type == "FISHEYE") {
      cameras_[i] = new FisheyeCamera;
    } else if (camera_type == "PINHOLE") {
      cameras_[i] = new PinholeCamera;
    } else {
      LOG(ERROR) << "Camera type: " << camera_type << " unsupported!";
      return false;
    }
    reader.read(reinterpret_cast<char*>(cameras_[i]->mutable_intrinsics()),
                9 * sizeof(double));
    reader.read(reinterpret_cast<char*>(cameras_[i]->mutable_extrinsics()),
                16 * sizeof(double));
    Eigen::Vector2i image_size;
    reader.read(reinterpret_cast<char*>(image_size.data()), 2 * sizeof(int));
    cameras_[i]->SetImageSize(image_size);
    std::vector<double> distortion_coeffs(
        cameras_[i]->GetDistortionCoeffs().size());
    reader.read(reinterpret_cast<char*>(distortion_coeffs.data()),
                distortion_coeffs.size() * sizeof(double));
    cameras_[i]->SetDistortionCoeffs(distortion_coeffs);
  }

  int num_points;
  reader.read(reinterpret_cast<char*>(&num_points), sizeof(int));
  points_.resize(num_points);
  for (int i = 0; i < num_points; i++) {
    points_[i] = new TrackedPoint;
    Eigen::Vector3d point3d;
    reader.read(reinterpret_cast<char*>(point3d.data()), sizeof(double) * 3);
    int num_observations;
    reader.read(reinterpret_cast<char*>(&num_observations), sizeof(int));
    for (int o = 0; o < num_observations; o++) {
      int camera_index;
      reader.read(reinterpret_cast<char*>(&camera_index), sizeof(int));
      Eigen::Vector3d observation;
      reader.read(reinterpret_cast<char*>(observation.data()),
                  sizeof(double) * 3);
      points_[i]->SetObservation(cameras_[camera_index], observation);
    }
  }
  LOG(INFO) << "Successfully loaded reconstruction containing "
            << cameras_.size() << " cameras and " << points_.size()
            << " points.";

  return true;
}

bool Reconstruction::Save(const std::string& filename) const {
  std::ofstream writer(filename, std::ios::out | std::ios::binary);
  if (!writer.is_open()) {
    LOG(ERROR) << "Unable to open: " << filename;
    return false;
  }
  const int num_cameras = cameras_.size();
  writer.write(reinterpret_cast<const char*>(&num_cameras), sizeof(int));
  std::unordered_map<const Camera*, int> camera_indices;
  for (int i = 0; i < cameras_.size(); i++) {
    const Camera* camera = cameras_[i];
    camera_indices[camera] = i;
    writer.write(Camera::TypeToString(camera->GetType()).c_str(), 7);
    writer.write(reinterpret_cast<const char*>(camera->intrinsics()),
                 9 * sizeof(double));
    writer.write(reinterpret_cast<const char*>(camera->extrinsics()),
                 16 * sizeof(double));
    writer.write(reinterpret_cast<const char*>(camera->GetImageSize().data()),
                 2 * sizeof(int));
    writer.write(reinterpret_cast<const char*>(camera->distortion_coeffs()),
                 camera->GetDistortionCoeffs().size() * sizeof(double));
  }
  const int num_points = points_.size();
  writer.write(reinterpret_cast<const char*>(&num_points), sizeof(int));
  for (int i = 0; i < points_.size(); i++) {
    const TrackedPoint* point = points_[i];
    writer.write(reinterpret_cast<const char*>(point->GetPoint().data()),
                 sizeof(double) * 3);
    const std::unordered_map<const Camera*, Eigen::Vector3d>& observations =
        point->GetObservations();
    const int num_observations = observations.size();
    writer.write(reinterpret_cast<const char*>(&num_observations), sizeof(int));

    for (const auto& observation : observations) {
      int camera_index = camera_indices[observation.first];
      CHECK_GE(camera_index, 0) << "Camera index invalid!";
      writer.write(reinterpret_cast<const char*>(&camera_index), sizeof(int));
      writer.write(reinterpret_cast<const char*>(observation.second.data()),
                   sizeof(double) * 3);
    }
  }
  return true;
}

void Reconstruction::SetPoints(const std::vector<TrackedPoint*> points) {
  points_ = points;
}
void Reconstruction::SetCameras(const std::vector<Camera*> cameras) {
  cameras_ = cameras;
}
void Reconstruction::AddCamera(Camera* camera) {
  CHECK_NOTNULL(camera);
  cameras_.push_back(camera);
}
void Reconstruction::AddPoint(TrackedPoint* point) {
  CHECK_NOTNULL(point);
  points_.push_back(point);
}

int Reconstruction::NumCameras() const { return cameras_.size(); }

int Reconstruction::NumPoints() const { return points_.size(); }

namespace {
void AddFrustumToMesh(Mesh* mesh, const Camera& camera) {
  const Eigen::Vector3f& lookat = camera.GetLookAt().cast<float>();
  const Eigen::Vector3f& up = camera.GetUpVector().cast<float>();
  const Eigen::Vector3f& left = -camera.GetRightVector().cast<float>();
  const Eigen::Vector3f& center = camera.GetPosition().cast<float>();

  static const float pyramid_height = 1.f;
  static const float pyramid_width = pyramid_height / 4;

  std::vector<VertexId> ids(5);
  ids[0] = mesh->AddVertex(center);
  ids[1] = mesh->AddVertex(center + (pyramid_height * lookat) +
                           pyramid_width * (up + left));
  ids[2] = mesh->AddVertex(center + (pyramid_height * lookat) +
                           pyramid_width * (up - left));
  ids[3] = mesh->AddVertex(center + (pyramid_height * lookat) +
                           pyramid_width * (-up + left));
  ids[4] = mesh->AddVertex(center + (pyramid_height * lookat) +
                           pyramid_width * (-up - left));
  mesh->AddTriangleFace(ids[0], ids[1], ids[2]);
  mesh->AddTriangleFace(ids[0], ids[2], ids[4]);
  mesh->AddTriangleFace(ids[0], ids[3], ids[1]);
  mesh->AddTriangleFace(ids[0], ids[4], ids[3]);
  mesh->AddTriangleFace(ids[1], ids[3], ids[2]);
  mesh->AddTriangleFace(ids[3], ids[4], ids[2]);
}

}  // namespace

void Reconstruction::SaveMesh(const std::string& filename) const {
  Mesh mesh;
  for (const auto& camera : cameras_) {
    AddFrustumToMesh(&mesh, *camera);
  }
  for (const auto& point : points_) {
    mesh.AddVertex(point->GetPoint().cast<float>());
  }
  mesh.Save(filename);
}

void Reconstruction::SaveTrajectoryMesh(const std::string& filename) const {
  Mesh mesh;
  for (const auto& camera : cameras_) {
    AddFrustumToMesh(&mesh, *camera);
  }
  mesh.Save(filename);
}

const Camera& Reconstruction::GetCamera(const int index) const {
  return *cameras_.at(index);
}

Camera* Reconstruction::GetCameraMutable(const int index) {
  return cameras_.at(index);
}

std::vector<Camera*> Reconstruction::FindSimilarViewpoints(const Camera* camera,
                                           const int angle_threshold) const {
  std::vector<Camera*> cameras;
  for (const auto cam : cameras_) {
    if (cam->GetLookAt().dot(camera->GetLookAt()) <
        cos(angle_threshold * M_PI / 180.0)) {
      cameras.push_back(cam);
    }
  }
  return cameras;
}

Mesh Reconstruction::CreateFrustumMesh() const {
  Mesh mesh;
  int i = 0; 
  for (const auto camera : cameras_) {
    i ++;
    Eigen::Vector3d position = camera->GetPosition();

    Eigen::Vector3f up = camera->GetUpVector().cast<float>();
    Eigen::Vector3f left = camera->GetRightVector().cast<float>();
    Eigen::Vector3f fwd = camera->GetLookAt().cast<float>();
    VertexId center = mesh.AddVertex(position.cast<float>());

    VertexId top_left =
        mesh.AddVertex(position.cast<float>() + (up / 2) + (left / 2) + fwd);
    VertexId bottom_left =
        mesh.AddVertex(position.cast<float>() - (up / 2) + (left / 2) + fwd);
    VertexId top_right =
        mesh.AddVertex(position.cast<float>() + (up / 2) - (left / 2) + fwd);
    VertexId bottom_right =
        mesh.AddVertex(position.cast<float>() - (up / 2) - (left / 2) + fwd);

    if (i % 10 == 0) {
      VertexId arrow_base_left =
          mesh.AddVertex(position.cast<float>() + fwd + (up / 2) - (left / 8));
      VertexId arrow_base_right =
          mesh.AddVertex(position.cast<float>() + fwd + (up / 2) + (left / 8));
      VertexId arrow_tip =
          mesh.AddVertex(position.cast<float>() + fwd + (up * 0.75f));
      VertexId arrow_edge_c_left =
          mesh.AddVertex(position.cast<float>() + fwd + (up * 0.65f) - (left / 8));
      VertexId arrow_edge_c_right =
          mesh.AddVertex(position.cast<float>() + fwd + (up * 0.65f) + (left / 8));
      VertexId arrow_edge_left =
          mesh.AddVertex(position.cast<float>() + fwd + (up * 0.65f) - (left / 6));
      VertexId arrow_edge_right =
          mesh.AddVertex(position.cast<float>() + fwd + (up * 0.65f) + (left / 6));
      mesh.AddTriangleFace(arrow_base_left, arrow_edge_c_left,
                           arrow_edge_c_right);
      mesh.AddTriangleFace(arrow_edge_c_right, arrow_base_right,
                           arrow_base_left);
      mesh.AddTriangleFace(arrow_edge_c_left, arrow_edge_left, arrow_tip);
      mesh.AddTriangleFace(arrow_edge_c_right, arrow_edge_c_left, arrow_tip);
      mesh.AddTriangleFace(arrow_edge_right, arrow_edge_c_right, arrow_tip);

      VertexId rarrow_base_left =
          mesh.AddVertex(position.cast<float>() + fwd + (-left / 2) - (up/ 8));
      VertexId rarrow_base_right =
          mesh.AddVertex(position.cast<float>() + fwd + (-left / 2) + (up / 8));
      VertexId rarrow_tip =
          mesh.AddVertex(position.cast<float>() + fwd + (-left * 0.75f));
      VertexId rarrow_edge_c_left =
          mesh.AddVertex(position.cast<float>() + fwd + (-left * 0.65f) - (up / 8));
      VertexId rarrow_edge_c_right =
          mesh.AddVertex(position.cast<float>() + fwd + (-left * 0.65f) + (up / 8));
      VertexId rarrow_edge_left =
          mesh.AddVertex(position.cast<float>() + fwd + (-left * 0.65f) - (up / 6));
      VertexId rarrow_edge_right =
          mesh.AddVertex(position.cast<float>() + fwd + (-left * 0.65f) + (up / 6));
      mesh.AddTriangleFace(rarrow_base_left, rarrow_edge_c_left,
                           rarrow_edge_c_right);
      mesh.AddTriangleFace(rarrow_edge_c_right, rarrow_base_right,
                           rarrow_base_left);
      mesh.AddTriangleFace(rarrow_edge_c_left, rarrow_edge_left, rarrow_tip);
      mesh.AddTriangleFace(rarrow_edge_c_right, rarrow_edge_c_left, rarrow_tip);
      mesh.AddTriangleFace(rarrow_edge_right, rarrow_edge_c_right, rarrow_tip);
      mesh.AddTriangleFace(center, top_right, top_left);
      mesh.AddTriangleFace(center, top_right, top_left);
      mesh.AddTriangleFace(center, bottom_right, top_right);
      mesh.AddTriangleFace(center, bottom_left, bottom_right);
      mesh.AddTriangleFace(center, top_left, bottom_left);
      mesh.AddTriangleFace(top_right, top_left, bottom_left);
      mesh.AddTriangleFace(bottom_left, bottom_right, top_right);
      mesh.AddTriangleFace(bottom_left, bottom_right, top_right);
    }
  }

  return mesh;
}

}  // namespace replay
