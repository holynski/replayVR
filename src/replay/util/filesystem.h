#include <iostream>

namespace replay {

// Given a path (e.g. /Users/person/test/image.png) will remove the filename,
// and return the deepest directory (/Users/person/test)
std::string GetDirectoryFromPath(const std::string& path);

// Given a path (e.g. /Users/person/test/image.png) will remove the directory,
// and return only the filename (image.png)
std::string GetFilenameFromPath(const std::string& path);

// Given a filename or path, returns the file extension (png, exe...)
// if it exists, otherwise returns an empty string
std::string GetExtension(const std::string& filename);

// Given a filename or path, returns the same string without the file extension,
// (image.png => image), (/home/test/file.txt => /home/test/file)
std::string RemoveExtension(const std::string& filename);

std::string JoinPath(const std::string& path1, const std::string& path2);

bool FileExists(const std::string& path);

bool DirectoryExists(const std::string& path);

bool CreateDirectory(const std::string& path);

std::vector<std::string> ListDirectory(const std::string& path,
                                       const std::string& must_contain = "");

}  // namespace replay
