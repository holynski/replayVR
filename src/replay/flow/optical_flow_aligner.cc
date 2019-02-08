
#include <replay/flow/optical_flow_aligner.h>
#include <replay/flow/visualization.h>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace replay {

OpticalFlowAligner::OpticalFlowAligner(const OpticalFlowType& type)
    : flow_(type) {}

cv::Mat OpticalFlowAligner::FlowWarp(const cv::Mat& input,
                                     const cv::Mat2f& flow) const {
  CHECK_EQ(input.size(), flow.size());
  cv::Mat2f map(flow.size());
  for (int y = 0; y < map.rows; ++y) {
    for (int x = 0; x < map.cols; ++x) {
      cv::Point2f f = flow.at<cv::Point2f>(y, x);
      map.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
    }
  }

  cv::Mat retval;
  cv::remap(input, retval, map, cv::Mat2f(), cv::INTER_NEAREST);
  return retval;
}

cv::Mat OpticalFlowAligner::Align(const cv::Mat& base,
                                  const cv::Mat& target) const {

  cv::imwrite("output/base.png", base);
  cv::imwrite("output/target.png", target);
  cv::Mat2f flow = flow_.ComputeFlow(base,target);
  cv::Mat retval = FlowWarp(base, flow);
  cv::imwrite("output/aligned.png", retval);
  cv::imwrite("output/flow.png", FlowToColor(flow));
  cv::imwrite("output/res_orig.png", retval - target);
  cv::imwrite("output/res_after.png", base - target);
  return retval;
}

}  // namespace replay
