#pragma once

#include <opencv2/opencv.hpp>
#include <replay/rendering/image_reprojector.h>

namespace replay {

class OpenGLContext;
class Camera;
class ImageReprojector;

class FlowFromReprojection {
 public:
  FlowFromReprojection(std::shared_ptr<OpenGLContext> context);

  cv::Mat2f Calculate(const Camera& src, const Camera& dst);

 private:
  ImageReprojector reprojector_;
};

}  // namespace replay
