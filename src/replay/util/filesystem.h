#include <iostream>

namespace replay {

std::string JoinPath(const std::string& path1, const std::string& path2);

bool FileExists(const std::string& path);

bool DirectoryExists(const std::string& path);

bool CreateDirectory(const std::string& path);

}  // namespace replay
