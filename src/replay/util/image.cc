#include "replay/util/image.h"
#include "replay/util/filesystem.h"

#include <FreeImage.h>
#include <glog/logging.h>

namespace replay {

Eigen::Vector2i GetImageSizeFromHeader(const std::string& filename) {
  CHECK(FileExists(filename)) << "File (" << filename << ") doesn't exist.";

  const FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename.c_str(), 0);

  CHECK_NE(format, FIF_UNKNOWN) << "Unrecognized image format.";

  FIBITMAP* fi_bitmap = FreeImage_Load(format, filename.c_str(), FIF_LOAD_NOPIXELS);
  if (fi_bitmap == nullptr) {
    LOG(FATAL) << "Image load failed.";
  }

  std::unique_ptr<FIBITMAP, decltype(&FreeImage_Unload)> data(
      fi_bitmap, &FreeImage_Unload);

  int width = FreeImage_GetWidth(data.get());
  int height = FreeImage_GetHeight(data.get());

  return Eigen::Vector2i(width, height);
}

}
