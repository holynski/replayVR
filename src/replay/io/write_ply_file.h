#pragma once

#include <glog/logging.h>
#include <tinyply/tinyply.h>
#include <fstream>
#include <string>
#include <vector>
#include "replay/io/write_ply_file.h"

#include "replay/geometry/mesh.h"

namespace replay {
// Writes the scene to disk as a PLY file.
bool WritePLYFile(const std::string& filename, const Mesh& mesh,
                  bool write_binary_file);

}  // namespace replay
