#include "replay/util/filesystem.h"
#include "replay/util/strings.h"

#include <dirent.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <iostream>

namespace replay {

namespace {}

std::string GetDirectoryFromPath(const std::string& path) {
  size_t found = path.find_last_of("/\\");
  return path.substr(0, found);
}

std::string GetFilenameFromPath(const std::string& path) {
  size_t found = path.find_last_of("/\\");
  return path.substr(found + 1);
}

std::string GetExtension(const std::string& filename) {
  size_t found = filename.find_last_of(".");
  return filename.substr(found + 1);
}

std::string RemoveExtension(const std::string& filename) {
  size_t found = filename.find_last_of(".");
  return filename.substr(0, found);
}

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

std::vector<std::string> ListDirectory(const std::string& path,
                                       const std::string& must_contain) {
  std::vector<std::string> files;
  DIR* dirp = opendir(path.c_str());
  struct dirent* dp;
  while ((dp = readdir(dirp)) != NULL) {
    std::string name = dp->d_name;
    if (must_contain.empty() || name.find(must_contain) != std::string::npos) {
      files.push_back(dp->d_name);
    }
  }
  closedir(dirp);
  std::sort(files.begin(), files.end());
  return files;
}

}  // namespace replay
