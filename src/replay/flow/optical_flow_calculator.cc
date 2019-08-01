#include <replay/flow/greedy_flow.h>
#include <replay/flow/optical_flow_calculator.h>
#include <replay/flow/visualization.h>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>

namespace replay {

OpticalFlowCalculator::OpticalFlowCalculator(
    const OpticalFlowType& type, std::shared_ptr<OpenGLContext> context)
    : type_(type), context_(context) {
  switch (type) {
    case DIS:
      flow_ = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
      break;
    case Simple:
      flow_ = cv::optflow::createOptFlow_SimpleFlow();
      break;
    case Farneback:
      flow_ = cv::optflow::createOptFlow_Farneback();
      break;
    case TVL1:
      flow_ = cv::optflow::createOptFlow_DualTVL1();
      break;
    case DeepFlow:
      flow_ = cv::optflow::createOptFlow_DeepFlow();
      break;
    case SparseToDense:
      flow_ = cv::optflow::createOptFlow_SparseToDense();
      break;
    case Greedy: {
      CHECK_NOTNULL(context_);
      GreedyFlow* greedy = new GreedyFlow(context);
      flow_ = cv::Ptr<cv::DenseOpticalFlow>(greedy);
    } break;
    default:
      LOG(FATAL) << "Optical flow type not implemented.";
  }
}

cv::Mat2f OpticalFlowCalculator::ComputeFlow(
    const cv::Mat& base, const cv::Mat& target,
    const cv::Mat2f& initialization) const {
  CHECK(!base.empty()) << "Images are empty!";

  cv::Mat base_gray = base;
  if (base.channels() != 1 && type_ != OpticalFlowType::Simple &&
      type_ != OpticalFlowType::Greedy) {
    LOG(ERROR) << "Converting!";
    cv::cvtColor(base, base_gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat target_gray = target;
  if (target.channels() != 1 && type_ != OpticalFlowType::Simple &&
      type_ != OpticalFlowType::Greedy) {
    cv::cvtColor(target, target_gray, cv::COLOR_BGR2GRAY);
  }

  cv::Mat2f flow = initialization.clone();
  if (flow.empty()) {
    flow = cv::Mat2f(base.size(), cv::Vec2f(0, 0));
  }

  flow_->calc(base_gray, target_gray, flow);

  return flow;
}

}  // namespace replay