#include "replay/io/write_depth_map.h"

#include <glog/logging.h>
#include <cereal/archives/portable_binary.hpp>
#include <fstream>   // NOLINT
#include <iostream>  // NOLINT
#include <string>

#include "replay/depth_map/depth_map.h"

namespace replay {

// Writes the depth map (and its confidence map) to disk.
bool WriteDepthMap(const std::string& depth_map_file,
                   const DepthMap& depth_map) {
  // Return false if the file cannot be opened for writing.
  std::ofstream depth_map_writer(depth_map_file,
                                 std::ios::out | std::ios::binary);
  if (!depth_map_writer.is_open()) {
    LOG(ERROR) << "Could not open the depth map file: " << depth_map_file
               << " for writing.";
    return false;
  }

  // Make sure that Cereal is able to finish executing before returning.
  {
    cereal::PortableBinaryOutputArchive output_archive(depth_map_writer);
    output_archive(depth_map);
  }
  depth_map_writer.close();
  return true;
}

bool WriteDepthMapNonCereal(const std::string& depth_map_file,
                            const DepthMap& depth_map) {
  std::ofstream depth_map_writer(depth_map_file,
                                 std::ios::out | std::ios::binary);
  if (!depth_map_writer.is_open()) {
    LOG(ERROR) << "Could not open the depth map file: " << depth_map_file
               << " for writing.";
    return false;
  }
  int rows = depth_map.Rows();
  int cols = depth_map.Cols();
  depth_map_writer.write(reinterpret_cast<const char*>(&(rows)), sizeof(int));
  depth_map_writer.write(reinterpret_cast<const char*>(&(cols)), sizeof(int));
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      float confidence = depth_map.GetConfidence(row, col);
      float depth = depth_map.GetDepth(row, col);
      depth_map_writer.write(reinterpret_cast<const char*>(&depth),
                             sizeof(float));
      depth_map_writer.write(reinterpret_cast<const char*>(&confidence),
                             sizeof(float));
    }
  }

  depth_map_writer.close();
  return true;
}

bool WriteDepthMapToTextFile(const std::string& depth_map_file,
                             const DepthMap& depth_map) {
  std::ofstream stream;
  stream.open(depth_map_file);
  for (int i = 0; i < depth_map.Rows(); i++) {
    for (int j = 0; j < depth_map.Cols(); j++) {
      if (i > 0 || j > 0) {
        stream << " ";
      }
      stream << depth_map.GetDepth(i, j);
    }
  }
  stream.close();
  return true;
}
}  // namespace replay
