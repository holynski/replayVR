#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>

#include "replay/util/lru_cache.h"
#include "replay/util/image.h"

namespace replay {

class DepthMap;

class DepthCache {
 public:
  // Constructor, takes the base folder in which the images are located and the
  // number of images to be stores in memory
  DepthCache(const std::string base_dir, const int cache_size);

  // Returns a single image by its filename. The returned cv::Mat will be empty
  // if the image does not exist.
  DepthMap Get(const std::string& filename) const;

 private:
  const std::string base_dir_;
  std::unique_ptr<LruCache<std::string, DepthMap>> cache_;
};
}  // namespace replay

