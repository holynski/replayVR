#include "replay/rendering/min_compositor.h"
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

MinCompositor::MinCompositor(std::shared_ptr<OpenGLContext> renderer)
    : renderer_(renderer), depth_renderer_(renderer_), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize renderer first!";
}

bool MinCompositor::Initialize(const Reconstruction& reconstruction,
                               const ImageCache& cache) {
  std::string vertex_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/min_composite.vs",
                                          &vertex_source));
  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/min_composite.fs",
                                          &fragment_source));
  fragment_source = ReplaceAll(fragment_source, "NUM_CAMERAS",
                               std::to_string(reconstruction.NumCameras()));
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(INFO) << "Compiling and linking shaders failed.";
    return false;
  }

  if (!depth_renderer_.Initialize()) {
    LOG(ERROR) << "Failed to initilize depth renderer";
    return false;
  }

  const int num_cameras = reconstruction.NumCameras();
  const Eigen::Vector2i image_size = reconstruction.GetCamera(0).GetImageSize();
  CHECK(renderer_->UseShader(shader_id_));
  renderer_->AllocateTextureArray("input_images", image_size.x(),
                                  image_size.y(), 3, num_cameras, true);

  renderer_->AllocateTextureArray("input_depths", image_size.x(),
                                  image_size.y(), 1, num_cameras, false,
                                  GL_FLOAT);

  DepthMap projection_matrix_tex(num_cameras * 4, 4);
  DepthMap position_tex(4, ((num_cameras + 1) / 2) * 2);
  for (int cam = 0; cam < num_cameras; cam++) {
    DepthMap depth_map;
    const Camera& camera = reconstruction.GetCamera(cam);
    depth_renderer_.GetDepthMap(camera, &depth_map);
    if (depth_map.Rows() != image_size.y() ||
        depth_map.Cols() != image_size.x()) {
      depth_map.Resize(image_size.y(), image_size.x());
    }
    CHECK(renderer_->UseShader(shader_id_));
    renderer_->UploadTextureToArray(depth_map, "input_depths", cam);

    cv::Mat image = cache.Get(camera.GetName());
    CHECK(!image.empty());
    ExposureAlignment::TransformImageExposure(image, camera.GetExposure(),
                                              Eigen::Vector3f(1, 1, 1), &image);
    if (image.rows != image_size.y() || image.cols != image_size.x()) {
      cv::resize(image, image, cv::Size(image_size.x(), image_size.y()));
    }
    renderer_->UploadTextureToArray(image, "input_images", cam);
    const Eigen::Vector3f position = camera.GetPosition().cast<float>();
    for (int k = 0; k < 3; k++) {
      position_tex.SetDepth(k, cam, position(k));
    }

    const Eigen::Matrix4f projection = camera.GetOpenGlMvpMatrix();
    for (int k = cam * 4; k < 4 + cam * 4; k++) {
      for (int j = 0; j < 4; j++) {
        projection_matrix_tex.SetDepth(k, j, projection(k % 4, j));
      }
    }
    renderer_->UploadTexture(position_tex, "position");
    renderer_->UploadTexture(projection_matrix_tex, "projection_matrix");
  }
  renderer_->HideWindow();
  is_initialized_ = true;
  return true;
}

bool MinCompositor::GetMinComposite(const Camera& viewpoint, cv::Mat* output) {
  CHECK(is_initialized_) << "Initialize MinCompositor first.";
  CHECK_NOTNULL(output);
  CHECK(renderer_->UseShader(shader_id_));

  Eigen::Vector3f current_position = viewpoint.GetPosition().cast<float>();
  renderer_->UploadShaderUniform(current_position, "virtual_camera_position");

  // Set the rendered viewpoint to correspond to the camera.
  if (!renderer_->SetViewpoint(viewpoint)) {
    return false;
  }

  renderer_->RenderToImage(output);
  return true;
}  // namespace replay

}  // namespace replay

// Stop uploading so many meshes!
// Write a version which is sequential -- but still uses the arrays. Avoid
// uploading images each time -- keep the array of images uploaded, but run in
// multiple passes. Fix the spherical projection code Make the windows for the
// other sequences Batch run
