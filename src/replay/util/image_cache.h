#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>

#include "replay/util/lru_cache.h"
#include "replay/util/image.h"

namespace replay {

class ImageCache {
 public:
  // Constructor, takes the base folder in which the images are located and the
  // number of images to be stores in memory
  ImageCache(const std::string base_dir, const int cache_size);

  // Returns a single image by its filename. The returned cv::Mat will be empty
  // if the image does not exist.
  const cv::Mat Get(const std::string& filename) const;

  // Returns the size of an image. Can avoid an expensive load if the image
  // isn't in memory.
  Eigen::Vector2i GetImageSize(const std::string& filename) const;

 private:
  const std::string base_dir_;
  std::unique_ptr<LruCache<std::string, cv::Mat>> cache_;
};
}  // namespace replay
