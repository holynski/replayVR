#ifndef REPLAY_IO_READ_DEPTH_MAP_H_
#define REPLAY_IO_READ_DEPTH_MAP_H_

#include <string>

namespace replay {
class DepthMap;

// Reads the depth map (and its confidence map) from disk.
bool ReadDepthMap(const std::string& depth_map_file, DepthMap* depth_map);

bool ReadDepthMapNonCereal(const std::string& depth_map_file,
                           DepthMap* depth_map);

bool ReadDepthMapFromTextFile(const std::string& depth_map_file,
                              DepthMap* depth_map);

}  // namespace replay

#endif  // REPLAY_IO_READ_DEPTH_MAP_H_
