#include "replay/io/read_depth_map.h"

#include <glog/logging.h>
#include <cereal/archives/portable_binary.hpp>
#include <fstream>   // NOLINT
#include <iostream>  // NOLINT
#include <string>

#include "replay/depth_map/depth_map.h"

namespace replay {

// Reads the depth map (and its confidence map) to disk.
bool ReadDepthMap(const std::string& depth_map_file, DepthMap* depth_map) {
  // Return false if the file cannot be opened for reading.
  std::ifstream depth_map_reader(depth_map_file,
                                 std::ios::in | std::ios::binary);
  if (!depth_map_reader.is_open()) {
    return false;
  }

  // Make sure that Cereal is able to finish executing before returning.
  {
    cereal::PortableBinaryInputArchive input_archive(depth_map_reader);
    input_archive(*depth_map);
  }
  depth_map_reader.close();
  return true;
}

bool ReadDepthMapNonCereal(const std::string& depth_map_file,
                           DepthMap* depth_map) {
  std::ifstream depth_map_reader(depth_map_file,
                                 std::ios::in | std::ios::binary);
  if (!depth_map_reader.is_open()) {
    return false;
  }
  int cols, rows;
  depth_map_reader.read(reinterpret_cast<char*>(&(rows)), sizeof(int));
  depth_map_reader.read(reinterpret_cast<char*>(&(cols)), sizeof(int));
  depth_map->Resize(rows, cols);
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      float confidence;
      float depth;
      depth_map_reader.read(reinterpret_cast<char*>(&(depth)), sizeof(float));
      depth_map_reader.read(reinterpret_cast<char*>(&(confidence)),
                            sizeof(float));
      depth_map->SetDepth(row, col, depth);
      depth_map->SetConfidence(row, col, confidence);
    }
  }
  return true;
}

bool ReadDepthMapFromTextFile(const std::string& depth_map_file,
                              DepthMap* depth_map) {
  CHECK_NOTNULL(depth_map);
  std::ifstream stream(depth_map_file);
  if (!stream.is_open()) {
    return false;
  }
  double width = 0;
  double height = 0;
  stream >> width;
  stream >> height;

  CHECK_GT(width, 0);
  CHECK_GT(height, 0);

  depth_map->Resize(height, width);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      double value;
      std::string token;
      stream >> value;
      depth_map->SetDepth(i, j, value);
    }
  }
  return true;
}

}  // namespace replay
