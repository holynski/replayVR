
#include <glog/logging.h>
#include <replay/rendering/opengl_context.h>
#include <opencv2/opencv.hpp>

namespace replay {

template <typename T>
SumAbsoluteDifference<T>::SumAbsoluteDifference(
    std::shared_ptr<OpenGLContext> renderer)
    : renderer_(renderer) {
  static const std::string shader_src_dir = REPLAY_SRC_DIR;
  CHECK(renderer->IsInitialized()) << "Initialize renderer first!";
  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/sad.fs",
                                          &fragment_source));
  CHECK(renderer_->CompileFullScreenShader(fragment_source, &shader_id_));
  renderer_->HideWindow();
}

template <typename T>
cv::Mat1f SumAbsoluteDifference<T>::GetDifference(const cv::Mat_<T>& image1,
                                                  const cv::Mat_<T>& image2,
                                                  const cv::Mat1b& mask1,
                                                  const cv::Mat1b& mask2,
                                                  const int window_size) {
  CHECK_EQ(image1.size(), image2.size());
  CHECK(renderer_->UseShader(shader_id_));
  renderer_->UploadShaderUniform(window_size, "window_size");
  Eigen::Vector2f image_size(image1.cols, image1.rows);
  renderer_->UploadShaderUniform(image_size, "image_size");
  if (mask1.empty()) {
    cv::Mat1b mask(image1.size(), 255);
    renderer_->UploadTexture(mask, "mask1");
  } else {
    renderer_->UploadTexture(mask1, "mask1");
  }
  if (mask2.empty()) {
    cv::Mat1b mask(image2.size(), 255);
    renderer_->UploadTexture(mask, "mask2");
  } else {
    renderer_->UploadTexture(mask2, "mask2");
  }

  renderer_->UploadTexture(image1, "image1");
  renderer_->UploadTexture(image2, "image2");
  renderer_->SetViewportSize(image_size.x(), image_size.y());
  cv::Mat1f retval(image1.size());
  renderer_->HideWindow();
  renderer_->RenderToImage(&retval);
  return retval;
}

}  // namespace replay
