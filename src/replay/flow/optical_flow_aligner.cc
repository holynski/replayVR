
#include <replay/flow/optical_flow_aligner.h>
#include <replay/flow/visualization.h>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace replay {

OpticalFlowAligner::OpticalFlowAligner(const OpticalFlowType& type,
                                       std::shared_ptr<OpenGLContext> context)
    : flow_(type, context) {}

cv::Mat OpticalFlowAligner::InverseWarp(const cv::Mat& source,
                                        const cv::Mat2f& flow_dest_to_source) {
  //CHECK_EQ(source.size(), flow_dest_to_source.size());
  cv::Mat2f map(flow_dest_to_source.size());
  cv::Mat remapped(flow_dest_to_source.size(), source.type());
  for (int y = 0; y < map.rows; ++y) {
    for (int x = 0; x < map.cols; ++x) {
      cv::Point2f f = flow_dest_to_source.at<cv::Point2f>(y, x);
      map.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
    }
  }

  cv::Mat retval;
  cv::remap(source, retval, map, cv::Mat2f(), cv::INTER_NEAREST);
  return retval;
}

cv::Mat OpticalFlowAligner::Align(const cv::Mat& base,
                                  const cv::Mat& target) const {
  cv::Mat2f flow = flow_.ComputeFlow(target, base);
  cv::Mat retval = InverseWarp(base, flow);
  return retval;
}

}  // namespace replay
