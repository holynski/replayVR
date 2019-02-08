#include "replay/util/image_cache.h"
#include "replay/util/filesystem.h"
#include "replay/util/image.h"
#include "replay/util/lru_cache.h"

#include <glog/logging.h>

namespace replay {

ImageCache::ImageCache(const std::string base_dir, const int cache_size)
    : base_dir_(base_dir),
      cache_(std::make_unique<LruCache<std::string, cv::Mat>>(cache_size)) {}

const cv::Mat ImageCache::Get(const std::string& filename) const {
  if (cache_->Exists(filename)) {
    return cache_->Get(filename);
  } else {
    cv::Mat image = cv::imread(JoinPath(base_dir_, filename));
    if (image.empty()) {
      LOG(ERROR) << "Failed to find file " << filename;
      return image;
    }
    cache_->Insert(filename, image);
    return cache_->Get(filename);
  }
}

Eigen::Vector2i ImageCache::GetImageSize(const std::string& filename) const {
  LOG(ERROR) << "Loading " << filename;
  return GetImageSizeFromHeader(JoinPath(base_dir_, filename));
}

}  // namespace replay
