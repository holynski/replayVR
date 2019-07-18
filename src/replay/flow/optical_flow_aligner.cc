
#include <replay/flow/optical_flow.h>
#include <replay/flow/optical_flow_aligner.h>
#include <replay/flow/visualization.h>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace replay {

OpticalFlowAligner::OpticalFlowAligner(const OpticalFlowType& type,
                                       std::shared_ptr<OpenGLContext> context)
    : flow_(type, context) {}

cv::Mat OpticalFlowAligner::Align(const cv::Mat& base,
                                  const cv::Mat& target) const {
  cv::Mat2f flow = flow_.ComputeFlow(target, base);
  cv::Mat retval = OpticalFlow::InverseWarp(base, flow);
  return retval;
}

}  // namespace replay
