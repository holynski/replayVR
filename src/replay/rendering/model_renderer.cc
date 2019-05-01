#include "replay/rendering/model_renderer.h"
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

ModelRenderer::ModelRenderer(std::shared_ptr<OpenGLContext> renderer)
    : renderer_(renderer), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize renderer first!";
}

bool ModelRenderer::Initialize() {
  std::string vertex_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/vertex_colored.vs",
                                          &vertex_source));
  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/vertex_colored.fs",
                                          &fragment_source));
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(INFO) << "Compiling and linking shaders failed.";
    return false;
  }

  renderer_->SetViewportSize(1000, 1000);

  CHECK(renderer_->UseShader(shader_id_));

  renderer_->HideWindow();
  is_initialized_ = true;
  return true;
}

bool ModelRenderer::RenderView(const Camera& viewpoint, cv::Mat* output) {
  CHECK(is_initialized_) << "Initialize ModelRenderer first.";
  CHECK_NOTNULL(output);
  CHECK(renderer_->UseShader(shader_id_));

  // Make sure the input and output have the same number of channels.
  const Eigen::Vector2i& image_size = viewpoint.GetImageSize();
  *output = cv::Mat3b(image_size.x(), image_size.y());

  // Set the rendered viewpoint to correspond to the camera.
  if (!renderer_->SetViewpoint(viewpoint)) {
    return false;
  }

  renderer_->RenderToImage(output);
  cv::resize(*output, *output, cv::Size(image_size.x(), image_size.y()));
  // renderer_->ShowWindow();
  // renderer_->Render();
  return true;
}

bool ModelRenderer::RenderView(const Eigen::Matrix4f& projection,
                               const Eigen::Vector2i& image_size,
                               cv::Mat* output) {
  CHECK(is_initialized_) << "Initialize ModelRenderer first.";
  CHECK_NOTNULL(output);
  CHECK(renderer_->UseShader(shader_id_));

  // Make sure the input and output have the same number of channels.
  *output = cv::Mat3b(image_size.x(), image_size.y());

  // Set the rendered viewpoint to correspond to the camera.
  if (!renderer_->SetProjectionMatrix(projection)) {
    return false;
  }

  renderer_->RenderToImage(output);
  cv::resize(*output, *output, cv::Size(image_size.x(), image_size.y()));
  return true;
}

}  // namespace replay
