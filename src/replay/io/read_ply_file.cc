#include "replay/io/read_ply_file.h"
#include <glog/logging.h>
#include <tinyply/tinyply.h>
#include <fstream>
#include <string>
#include <vector>

#include "replay/geometry/mesh.h"

namespace replay {
bool ReadPLYFile(const std::string& filename, Mesh* mesh,
                 std::vector<std::string>* comments) {
  std::ifstream stream(filename, std::ios::binary);
  if (!stream.is_open()) {
    return false;
  }
  tinyply::PlyFile ply_file(stream);
  if (comments != nullptr) {
    *comments = ply_file.comments;
  }
  std::vector<float> vertex_stream;
  ply_file.request_properties_from_element("vertex", {"x", "y", "z"},
                                           vertex_stream);
  std::vector<int> faces;
  ply_file.request_properties_from_element("face", {"vertex_indices"}, faces,
                                           3);
  std::vector<float> uvs;
  ply_file.request_properties_from_element("vertex", {"texture_u", "texture_v"},
                                           uvs);

  std::vector<uint8_t> color;
  ply_file.request_properties_from_element("vertex", {"red", "green", "blue"},
                                           color);
  ply_file.read(stream);
  for (int i = 0; i < vertex_stream.size(); i += 3) {
    mesh->AddVertex(Eigen::Vector3f(vertex_stream[i], vertex_stream[i + 1],
                                    vertex_stream[i + 2]));
  }
  for (int i = 0; i < faces.size(); i += 3) {
    mesh->AddTriangleFace(faces[i], faces[i + 1], faces[i + 2]);
  }
  for (int i = 0; i < uvs.size(); i += 2) {
    mesh->SetVertexUV(i / 2, uvs[i], 1.f - uvs[i + 1]);
  }
  for (int i = 0; i < color.size(); i += 3) {
    mesh->SetVertexColor(i / 3, Eigen::Vector3f(color[i], color[i+1], color[i+2]) / 255.0);
  }
  stream.close();

  LOG(INFO) << "Mesh loaded with " << mesh->NumVertices() << " vertices and "
            << mesh->NumTriangleFaces() << " triangles.";
  const int num_isolated_vertices = mesh->RemoveIsolatedVertices();
  mesh->CollectGarbage();
  LOG(INFO) << "Removed " << num_isolated_vertices
            << " isolated vertices from the mesh.";

  return true;
}
}  // namespace replay
