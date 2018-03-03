#include "replay/io/write_ply_file.h"
#include <glog/logging.h>
#include <tinyply/tinyply.h>
#include <fstream>
#include <string>
#include <vector>

#include "replay/mesh/mesh.h"

namespace replay {
// Writes the scene to disk as a PLY file.
bool WritePLYFile(const std::string& filename, const Mesh& mesh,
                  bool write_binary_file) {
  std::ofstream file(filename, std::ios::binary);
  std::ostringstream stream;
  // Return false if the file cannot be opened for writing.
  if (!file.is_open()) {
    return false;
  }
  tinyply::PlyFile ply_file;
  std::vector<float> vertices(mesh.NumVertices() * 3);
  for (int i = 0; i < mesh.NumVertices(); i++) {
    Eigen::Map<Eigen::Vector3f> vertex_position(vertices.data() + i * 3);
    vertex_position = mesh.VertexPosition(i);
  }
  ply_file.add_properties_to_element("vertex", {"x", "y", "z"}, vertices);
  std::vector<int> faces;
  faces.resize(mesh.NumTriangleFaces() * 3);
  for (int i = 0; i < mesh.NumTriangleFaces(); i++) {
    const Eigen::Matrix<TriangleFaceId, 3, 1> triangle =
        mesh.GetVertexIdsForTriangleFace(i);
    faces[i * 3] = triangle[0];
    faces[i * 3 + 1] = triangle[1];
    faces[i * 3 + 2] = triangle[2];
  }
  ply_file.add_properties_to_element("face", {"vertex_indices"}, faces, 3,
                                     tinyply::PlyProperty::Type::UINT8);
  std::vector<float> uvs;
  if (mesh.HasUVs()) {
    uvs.resize(mesh.NumVertices() * 2);
    for (int i = 0; i < mesh.NumVertices(); i++) {
      const Eigen::Vector2f& uv = mesh.VertexUV(i);
      uvs[i * 2] = uv[0];
      uvs[i * 2 + 1] = 1.f - uv[1];
    }
    ply_file.add_properties_to_element("vertex", {"texture_u", "texture_v"},
                                       uvs);
  }
  ply_file.write(stream, write_binary_file);
  file << stream.str();
  file.close();
  return true;
}
}  // namespace replay
