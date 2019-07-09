#include "replay/rendering/equirect_reprojector.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "replay/camera/camera.h"
#include "replay/depth_map/depth_map.h"
#include "replay/geometry/mesh.h"
#include "replay/rendering/depth_map_renderer.h"
#include "replay/sfm/reconstruction.h"
#include "replay/util/image_cache.h"

namespace replay {

static const std::string shader_src_dir = REPLAY_SRC_DIR;

EquirectReprojector::EquirectReprojector(std::shared_ptr<OpenGLContext> context)
    : context_(context) {
  CHECK(context->IsInitialized()) << "Initialize renderer first!";

  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/to_equirect.fs",
                                          &fragment_source));
  CHECK(context_->CompileFullScreenShader(fragment_source, &shader_id_to_))
      << "Compiling and linking shaders failed.";

  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/from_equirect.fs",
                                          &fragment_source));
  CHECK(context_->CompileFullScreenShader(fragment_source, &shader_id_from_))
      << "Compiling and linking shaders failed.";
}

bool EquirectReprojector::ProjectToEquirect(const cv::Mat& image, const Camera& camera, 
                                    cv::Mat* reprojected) {
  CHECK(context_->UseShader(shader_id_to_));

  if (camera.GetImageSize().norm() == 0) {
    LOG(ERROR) << "Camera has zero image size.";
    return false;
  }
  CHECK(context_->UseShader(shader_id_to_));
  Eigen::Matrix4f intrinsics = Eigen::Matrix4f::Identity();
  intrinsics.block(0,0,3,3) = camera.GetIntrinsicsMatrix().cast<float>();
  Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
  rotation.block(0,0,3,3) = camera.GetRotation().cast<float>();
  context_->UploadShaderUniform(intrinsics, "intrinsics");
  context_->UploadShaderUniform(rotation, "rotation");
  //context_->UploadShaderUniform(
      //Eigen::Matrix4f
          //(intrinsics * camera.GetExtrinsics().cast<float>()),
      //"input_projection_matrix");
  const Eigen::Vector3f position = camera.GetPosition().cast<float>();
  context_->UploadShaderUniform(position, "input_position");

  if (image.empty()) {
    LOG(ERROR) << "Image is empty!";
    return false;
  }
  CHECK(context_->UseShader(shader_id_to_));
  context_->UploadTexture(image, "input_image");

  //const Eigen::Vector2i& image_size = camera.GetImageSize();
  //const Eigen::Vector2f& image_size_float = image_size.cast<float>();
  context_->UploadShaderUniform(Eigen::Vector2f(4800,2700), "image_size");
  *reprojected = cv::Mat3b(2700,4800);
  context_->SetViewportSize(4800, 2700);

  context_->RenderToImage(reprojected);
  return true;
}

bool EquirectReprojector::ProjectFromEquirect(const cv::Mat& image,
                                              const Camera& camera,
                                              cv::Mat* reprojected) {

  CHECK(context_->UseShader(shader_id_from_));

  if (camera.GetImageSize().norm() == 0) {
    LOG(ERROR) << "Camera has zero image size.";
    return false;
  }
  CHECK(context_->UseShader(shader_id_from_));
  Eigen::Matrix4f intrinsics = Eigen::Matrix4f::Identity();
  intrinsics.block(0,0,3,3) = camera.GetIntrinsicsMatrix().cast<float>();
  Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
  rotation.block(0,0,3,3) = camera.GetRotation().cast<float>();
  context_->UploadShaderUniform(intrinsics, "intrinsics");
  context_->UploadShaderUniform(rotation, "rotation");
  //context_->UploadShaderUniform(
      //Eigen::Matrix4f
          //(intrinsics * camera.GetExtrinsics().cast<float>()),
      //"input_projection_matrix");
  const Eigen::Vector3f position = camera.GetPosition().cast<float>();
  context_->UploadShaderUniform(position, "input_position");

  if (image.empty()) {
    LOG(ERROR) << "Image is empty!";
    return false;
  }
  CHECK(context_->UseShader(shader_id_from_));
  context_->UploadTexture(image, "input_image");

  const Eigen::Vector2i& image_size = camera.GetImageSize();
  const Eigen::Vector2f& image_size_float = image_size.cast<float>();
  context_->UploadShaderUniform(image_size_float, "image_size");
  *reprojected = cv::Mat3b(image_size.y(), image_size.x());
  context_->SetViewportSize(image_size.x(), image_size.y());

  context_->RenderToImage(reprojected);
  return true;
}

}  // namespace replay
