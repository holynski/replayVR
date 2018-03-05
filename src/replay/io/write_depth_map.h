#ifndef REPLAY_IO_WRITE_DEPTH_MAP_H_
#define REPLAY_IO_WRITE_DEPTH_MAP_H_

#include <string>

namespace replay {
class DepthMap;

// Writes the depth map (and its confidence map) to disk.
bool WriteDepthMap(const std::string& depth_map_file,
                   const DepthMap& depth_map);

bool WriteDepthMapNonCereal(const std::string& depth_map_file,
                            const DepthMap& depth_map);

bool WriteDepthMapToTextFile(const std::string& depth_map_file,
                             const DepthMap& depth_map);

}  // namespace replay

#endif  // REPLAY_IO_WRITE_DEPTH_MAP_H_
