#pragma once

#include "replay/camera/camera.h"
#include "replay/depth_map/depth_map.h"
#include "replay/geometry/mesh.h"
#include "replay/rendering/depth_map_renderer.h"
#include "replay/rendering/opengl_context.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"

namespace replay {

class ImageReprojector {
 public:
  explicit ImageReprojector(std::shared_ptr<OpenGLContext> context);

  bool SetSourceCamera(const Camera& camera);
  bool SetImage(const cv::Mat& image);

  bool Reproject(const Camera& camera, cv::Mat* reprojected,
                 const float depth_tolerance = 0.005);

 private:
  std::shared_ptr<OpenGLContext> context_;
  DepthMapRenderer depth_renderer_;
  int shader_id_;
  bool camera_set_;
  bool image_set_;
  cv::Mat input_image_;
};
}  // namespace replay
