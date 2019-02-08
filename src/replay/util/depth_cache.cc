#include "replay/util/depth_cache.h"
#include "replay/depth_map/depth_map.h"
#include "replay/util/filesystem.h"
#include "replay/util/image.h"
#include "replay/util/lru_cache.h"

#include <glog/logging.h>

namespace replay {

DepthCache::DepthCache(const std::string base_dir, const int cache_size)
    : base_dir_(base_dir),
      cache_(std::make_unique<LruCache<std::string, DepthMap>>(cache_size)) {}

DepthMap DepthCache::Get(const std::string& filename) const {
  if (cache_->Exists(filename)) {
    return cache_->Get(filename);
  } else {
    DepthMap image(filename);
    if (image.Rows() == 0 || image.Cols() == 0) {
      LOG(ERROR) << "Failed to find file " << filename;
      return image;
    }
    cache_->Insert(filename, image);
    return cache_->Get(filename);
  }
}

}  // namespace replay

