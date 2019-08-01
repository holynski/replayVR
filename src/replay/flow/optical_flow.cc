#include <replay/flow/optical_flow.h>
#include <opencv2/opencv.hpp>

namespace replay {

namespace OpticalFlow {

cv::Mat2f Scale(const cv::Mat2f& flow, const float scale) {
  cv::Mat2f output_flow;
  cv::resize(flow, output_flow, cv::Size(), scale, scale, cv::INTER_AREA);
  output_flow /= scale;
  return output_flow;
}

cv::Mat InverseWarp(const cv::Mat& source,
                    const cv::Mat2f& flow_dest_to_source) {
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

}  // namespace OpticalFlow

}  // namespace replay
