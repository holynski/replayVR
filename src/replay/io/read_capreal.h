#pragma once

#include <string>

namespace replay {
class Reconstruction;
class ImageCache;

bool ReadCapreal(const std::string& filename, const ImageCache& cache,
                 Reconstruction* recon);
}
