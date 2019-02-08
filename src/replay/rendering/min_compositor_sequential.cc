#include "replay/rendering/min_compositor_sequential.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "replay/camera/camera.h"
#include "replay/depth_map/depth_map.h"
#include "replay/multiview/exposure_alignment.h"
#include "replay/rendering/depth_map_renderer.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"
#include "replay/util/strings.h"

namespace replay {

static const std::string shader_src_dir = REPLAY_SRC_DIR;

MinCompositorSequential::MinCompositorSequential(
    std::shared_ptr<OpenGLContext> renderer)
    : renderer_(renderer), depth_renderer_(renderer_), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize renderer first!";
}

bool MinCompositorSequential::Initialize() {
  std::string vertex_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/min_composite.vs",
                                          &vertex_source));
  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(
      shader_src_dir + "/min_composite_seq.fs", &fragment_source));
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(INFO) << "Compiling and linking shaders failed.";
    return false;
  }

  if (!depth_renderer_.Initialize()) {
    LOG(ERROR) << "Failed to initilize depth renderer";
    return false;
  }

  CHECK(renderer_->UseShader(shader_id_));

  renderer_->HideWindow();
  is_initialized_ = true;
  return true;
}

bool MinCompositorSequential::GetMinComposite(
    const Reconstruction& reconstruction, const ImageCache& cache,
    const Camera& viewpoint, cv::Mat* output) {
  CHECK(is_initialized_) << "Initialize MinCompositorSequential first.";
  CHECK_NOTNULL(output);
  CHECK(renderer_->UseShader(shader_id_));

  Eigen::Vector3f current_position = viewpoint.GetPosition().cast<float>();
  renderer_->UploadShaderUniform(current_position, "virtual_camera_position");

  const int num_cameras = reconstruction.NumCameras();
  const Eigen::Vector2i image_size = reconstruction.GetCamera(0).GetImageSize();
  CHECK(renderer_->UseShader(shader_id_));

  *output = cv::Mat3b(image_size.y(), image_size.x(), cv::Vec3b(255, 255, 255));
  if (!renderer_->SetViewpoint(viewpoint)) {
    return false;
  }

  for (int cam = 0; cam < num_cameras; cam++) {
    renderer_->UploadTexture(*output, "min_composite");
    const Camera& camera = reconstruction.GetCamera(cam);
    cv::Mat image = cache.Get(camera.GetName()).clone();
    CHECK(!image.empty());
    ExposureAlignment::TransformImageExposure(image, camera.GetExposure(),
                                              Eigen::Vector3f(1, 1, 1), &image);
    DepthMap depth_map;
    depth_renderer_.GetDepthMap(camera, &depth_map);
    CHECK(renderer_->UseShader(shader_id_));
    renderer_->UploadTexture(image, "input_image");
    renderer_->UploadTexture(depth_map, "input_depth");
    const Eigen::Vector3f position = camera.GetPosition().cast<float>();
    const Eigen::Matrix4f projection = camera.GetOpenGlMvpMatrix();
    renderer_->UploadShaderUniform(position, "position");
    renderer_->UploadShaderUniform(projection, "projection_matrix");
    renderer_->SetViewpoint(viewpoint);
    renderer_->RenderToImage(output);
  }

  return true;
}  // namespace replay

}  // namespace replay

// Stop uploading so many meshes!
// Write a version which is sequential -- but still uses the arrays. Avoid
// uploading images each time -- keep the array of images uploaded, but run in
// multiple passes. Fix the spherical projection code Make the windows for the
// other sequences Batch run
