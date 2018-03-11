#pragma once

#include <replay/mesh/mesh.h>
#include <replay/io/stream_reader.h>
#include <Eigen/Sparse>
#include <iostream>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>

namespace replay {

class MeshProjectionParser {
  public:
  std::vector<Mesh> Parse(const AVSphericalMesh& metadata,
                                 const uint32_t encoding);
  private:
  Mesh ParseMesh();
  std::unique_ptr<StreamReader> reader_;
};

}  // namespace replay
