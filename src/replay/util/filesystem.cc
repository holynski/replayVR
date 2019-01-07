#include "replay/util/filesystem.h"

#include <glog/logging.h>
#include <sys/stat.h>
#include <iostream>

namespace replay {

std::string JoinPath(const std::string& path1, const std::string& path2) {
  return path1 + (path1[path1.length() - 1] == '/' ? "" : "/") + path2;
}

bool FileExists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

bool DirectoryExists(const std::string& path) {
  LOG(FATAL) << "Not implemented";
}

bool CreateDirectory(const std::string& path) {
  LOG(FATAL) << "Not implemented";
}

}  // namespace replay
