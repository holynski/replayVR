#pragma once

#include <glog/logging.h>
#include <replay/rendering/opengl_context.h>
#include <opencv2/opencv.hpp>

namespace replay {

namespace {
static const std::string shader_src_dir = REPLAY_SRC_DIR;
}

template <typename T>
class FuzzyMinDifference {
 public:
  FuzzyMinDifference(std::shared_ptr<OpenGLContext> renderer);
  cv::Mat_<T> GetDifference(const cv::Mat_<T>& image1,
                            const cv::Mat_<T>& image2,
                            const int window_size = 10);

  std::shared_ptr<OpenGLContext> renderer_;
  int shader_id_;
  bool is_initialized_;
};

template <typename T>
FuzzyMinDifference<T>::FuzzyMinDifference(
    std::shared_ptr<OpenGLContext> renderer)
    : renderer_(renderer) {
  CHECK(renderer->IsInitialized()) << "Initialize renderer first!";
  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(
      shader_src_dir + "/fuzzy_difference.fs", &fragment_source));
  CHECK(renderer_->CompileFullScreenShader(fragment_source, &shader_id_));
  renderer_->HideWindow();
}

template <typename T>
cv::Mat_<T> FuzzyMinDifference<T>::GetDifference(const cv::Mat_<T>& image1,
                                                 const cv::Mat_<T>& image2,
                                                 const int window_size) {
  CHECK_EQ(image1.size(), image2.size());
  CHECK(renderer_->UseShader(shader_id_));
  renderer_->UploadShaderUniform(window_size, "window_size");
  Eigen::Vector2f image_size(image1.cols, image1.rows);
  renderer_->UploadShaderUniform(image_size, "image_size");
  renderer_->UploadTexture(image1, "image1");
  renderer_->UploadTexture(image2, "image2");
  renderer_->SetViewportSize(image_size.x(), image_size.y());
  cv::Mat_<T> retval(image1.size());
  renderer_->HideWindow();
  renderer_->RenderToImage(&retval);
  double minval, maxval;
  cv::minMaxLoc(retval, &minval, &maxval);
  return retval;
}

}  // namespace replay
