#pragma once

#include <string>

namespace replay {
class Reconstruction;
class ImageCache;

bool ReadBundler(const std::string& bundler_filename, const std::string& image_list,
                 const ImageCache& cache, Reconstruction* recon);
}
