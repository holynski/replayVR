#include "replay/rendering/image_reprojector.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "replay/camera/camera.h"
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/rendering/depth_map_renderer.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"

namespace replay {

static const std::string shader_src_dir = REPLAY_SRC_DIR;

ImageReprojector::ImageReprojector(std::shared_ptr<OpenGLContext> context)
    : context_(context),
      depth_renderer_(context),
      camera_set_(false),
      image_set_(false) {
  CHECK(context->IsInitialized()) << "Initialize renderer first!";

  std::string vertex_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/reprojector.vs",
                                          &vertex_source));
  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/reprojector.fs",
                                          &fragment_source));
  CHECK(context_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_))
      << "Compiling and linking shaders failed.";
  CHECK(depth_renderer_.Initialize());
}

bool ImageReprojector::SetSourceCamera(const Camera& camera) {
  if (camera.GetImageSize().norm() == 0) {
    LOG(ERROR) << "Camera has zero image size.";
    return false;
  }
  CHECK(context_->UseShader(shader_id_));
  context_->UploadShaderUniform(camera.GetOpenGlMvpMatrix(),
                                "input_projection_matrix");
  const Eigen::Vector3f position = camera.GetPosition().cast<float>();
  context_->UploadShaderUniform(position, "input_position");
  DepthMap depth_map;
  depth_renderer_.GetDepthMap(camera, &depth_map);
  if (depth_map.Rows() % 2 == 1 || depth_map.Cols() % 2 == 1) {
    depth_map.Resize(depth_map.Rows() - (depth_map.Rows() % 2),
                     depth_map.Rows() - (depth_map.Cols() % 2));
  }
  CHECK(context_->UseShader(shader_id_));
  // if (camera_set_) {
  // context_->UpdateTexture(depth_map, "input_depth");

  //} else {
  context_->UploadTexture(depth_map, "input_depth");
  camera_set_ = true;
  //}
  return true;
}

bool ImageReprojector::SetImage(const cv::Mat& image) {
  if (image.empty()) {
    LOG(ERROR) << "Image is empty!";
    return false;
  }
  CHECK(context_->UseShader(shader_id_));
  // if (image_set_) {
  // context_->UpdateTexture(image, "input_image");
  //} else {
  context_->UploadTexture(image, "input_image");
  image_set_ = true;
  input_image_ = image;
  //}
  return true;
}

bool ImageReprojector::Reproject(const Camera& camera, cv::Mat* reprojected,
                                 const float depth_tolerance) {
  CHECK(context_->UseShader(shader_id_));
  context_->UploadShaderUniform(depth_tolerance, "max_depth_error_percent");
  const Eigen::Vector2i& image_size = camera.GetImageSize();
  *reprojected =
      cv::Mat(cv::Size(image_size.x(), image_size.y()), input_image_.type());
  if (!context_->SetViewpoint(camera)) {
    return false;
  }
  context_->RenderToImage(reprojected);
  return true;
}

}  // namespace replay
