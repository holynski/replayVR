
#include "replay/flow/greedy_flow.h"
#include "replay/flow/visualization.h"
#include <glog/logging.h>
#include "replay/rendering/opengl_context.h"

namespace replay {

GreedyFlow::GreedyFlow(std::shared_ptr<OpenGLContext> context,
                       const int window_size)
    : context_(context), window_size_(window_size) {
  static const std::string shader_src_dir = REPLAY_SRC_DIR;
  CHECK(context_->IsInitialized()) << "Initialize renderer first!";
  std::string fragment_source;
  CHECK(OpenGLContext::ReadShaderFromFile(shader_src_dir + "/greedy_flow.fs",
                                          &fragment_source));
  CHECK(context_->CompileFullScreenShader(fragment_source, &shader_id_));
  context_->HideWindow();
}

void GreedyFlow::SetWindowSize(const int window_size) {
  window_size_ = window_size;
}

void GreedyFlow::calc(cv::InputArray I0, cv::InputArray I1,
                      cv::InputOutputArray flow) {
  //CHECK_EQ(I0.size(), I1.size());
  CHECK_EQ(I0.size(), flow.size());
  CHECK(context_->UseShader(shader_id_));
  context_->UploadShaderUniform(window_size_, "window_size");
  Eigen::Vector2f image_size(I0.cols(), I0.rows());
  context_->UploadShaderUniform(image_size, "image_size");

  //flow.getMat().setTo(0, flow.getMat() == FLT_MAX);
  context_->UploadTexture(flow.getMat(), "initial_flow");
  context_->UploadTexture(I0.getMat(), "image1");
  context_->UploadTexture(I1.getMat(), "image2");
  context_->SetViewportSize(image_size.x(), image_size.y());
  cv::Mat1f retval(I0.size());
  context_->HideWindow();
  cv::Mat flowMat = flow.getMat();
  context_->RenderToImage(&flowMat);
}

void GreedyFlow::collectGarbage() {
  LOG(ERROR)
      << "Garbage collection not implemented...there may be memory leaks.";
}

}  // namespace replay
