#include "replay/rendering/image_based_renderer.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "replay/camera/camera.h"
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/multiview/exposure_alignment.h"
#include "replay/rendering/depth_map_renderer.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"
#include "replay/util/strings.h"

namespace replay {

static const std::string shader_src_dir = REPLAY_SRC_DIR;
static const int kMaxNumViewsInRenderer = 350;

ImageBasedRenderer::ImageBasedRenderer(std::shared_ptr<OpenGLContext> renderer)
    : renderer_(renderer), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize renderer first!";
}

bool ImageBasedRenderer::Initialize(const Reconstruction& reconstruction,
                                    const ImageCache& cache, const Mesh& mesh) {
  std::string vertex_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/top3_ibr.vs",
                                          &vertex_source));
  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/min_composite.fs",
                                          &fragment_source));
  fragment_source =
      ReplaceAll(fragment_source, "NUM_CAMERAS", std::to_string(reconstruction.NumCameras()));
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(INFO) << "Compiling and linking shaders failed.";
    return false;
  }

  const Camera& base_cam_ = reconstruction.GetCamera(0);
  const Eigen::Vector2i& image_size = base_cam_.GetImageSize();
  renderer_->SetViewportSize(image_size.x(), image_size.y());

  CHECK(renderer_->UseShader(shader_id_));
  const int num_cameras = reconstruction.NumCameras();

  renderer_->UploadShaderUniform(num_cameras, "num_cameras");
  renderer_->AllocateTextureArray("input_images", image_size.x(),
                                  image_size.y(), 3, num_cameras);
  renderer_->AllocateTextureArray("input_depths", image_size.x(),
                                  image_size.y(), 1, num_cameras, false,
                                  GL_FLOAT);

  std::vector<Eigen::Vector3f> centers;
  std::vector<Eigen::Matrix4f> projections;
  const int num_views = std::min<int>(num_cameras, kMaxNumViewsInRenderer);
  DepthMap input_positions(num_views + (num_views % 2), 4);
  for (int i = 0; i < num_views; i++) {
    const Camera& camera = reconstruction.GetCamera(i);
    cv::Mat image = cache.Get(camera.GetName());
    CHECK(!image.empty());
    cv::resize(image, image, cv::Size(image_size.x(), image_size.y()));
    // cv::flip(image, image, 0);
    ExposureAlignment::TransformImageExposure(image, camera.GetExposure(),
                                              Eigen::Vector3f(1, 1, 1), &image);
    // cv::flip(image,image,1);
    renderer_->UploadTextureToArray(image, "input_images", i);
    const Eigen::Vector3d position = camera.GetPosition();
    input_positions.SetDepth(0, i, position[0]);
    input_positions.SetDepth(1, i, position[1]);
    input_positions.SetDepth(2, i, position[2]);
    projections.push_back(camera.GetOpenGlMvpMatrix());
  }

  renderer_->UploadTexture(input_positions, "input_positions");
  renderer_->UploadShaderUniform(projections, "input_projection_matrices");

  mesh_id_ = renderer_->UploadMesh(mesh);
  renderer_->BindMesh(mesh_id_);

  DepthMapRenderer depth_renderer(renderer_);
  depth_renderer.Initialize();
  for (int i = 0; i < num_views; i++) {
    DepthMap depth_map;
    const Camera& camera = reconstruction.GetCamera(i);
    depth_renderer.GetDepthMap(camera, &depth_map);
    depth_map.Resize(image_size.y(), image_size.x());
    CHECK(renderer_->UseShader(shader_id_));
    renderer_->UploadTextureToArray(depth_map, "input_depths", i);
  }

  renderer_->HideWindow();
  is_initialized_ = true;
  return true;
}

bool ImageBasedRenderer::RenderView(const Camera& viewpoint, cv::Mat* output,
                                    int id) {
  CHECK(is_initialized_) << "Initialize ImageBasedRenderer first.";
  CHECK_NOTNULL(output);
  CHECK(renderer_->UseShader(shader_id_));

  // Make sure the input and output have the same number of channels.
  const Eigen::Vector2i& image_size = viewpoint.GetImageSize();
  *output = cv::Mat3b(image_size.x(), image_size.y());

  // Set the rendered viewpoint to correspond to the camera.
  if (!renderer_->SetViewpoint(viewpoint)) {
    return false;
  }

  // Upload the current viewpoint position for IBR cost evaluation.
  Eigen::Vector3f position = viewpoint.GetPosition().cast<float>();
  renderer_->UploadShaderUniform(position, "virtual_camera_position");
  renderer_->UploadShaderUniform(id, "cam_id");

  CHECK(renderer_->BindMesh(mesh_id_)) << "Mesh binding failed.";
  renderer_->RenderToImage(output);
  // renderer_->ShowWindow();
  // renderer_->Render();
  return true;
}

}  // namespace replay
