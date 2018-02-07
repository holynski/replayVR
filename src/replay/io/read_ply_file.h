#pragma once

#include <string>
#include <vector>

namespace replay {
class Mesh;

bool ReadPLYFile(const std::string& filename, Mesh* mesh,
                 std::vector<std::string>* comments = nullptr);

}  // namespace replay
