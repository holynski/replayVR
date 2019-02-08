#pragma once

#include <FreeImage.h>
#include <Eigen/Core>

namespace replay {

// Reads the image file header to determine size (without reading the full image
// data)
Eigen::Vector2i GetImageSizeFromHeader(const std::string& filename);

}
