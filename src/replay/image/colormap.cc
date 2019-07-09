#include <glog/logging.h>
#include <replay/image/colormap.h>
#include <opencv2/opencv.hpp>

namespace replay {

namespace {
int ColormapToOpenCv(const Colormap cm) {
  switch (cm) {
    case Colormap::Jet:
      return cv::COLORMAP_JET;
    default:
      LOG(FATAL) << "Colormap not supported.";
  }
  return 0;
}
}  // namespace

cv::Mat3b FloatToColor(const cv::Mat1f& image, const Colormap cm) {
  cv::Mat3b output(image.size());
  double min;
  double max;
  cv::minMaxLoc(image, &min, &max);

  cv::Mat1b gray;
  image.convertTo(gray, CV_8U, 255 * 1.0/(max - min), 255 * (-min / (max - min)));

  cv::applyColorMap(gray, output, ColormapToOpenCv(cm));

  return output;
}

cv::Mat3b FloatToColor(const cv::Mat1f& image, const float min_scale,
                       const float max_scale, const Colormap cm) {
  cv::Mat3b output(image.size());

  cv::Mat1b gray;
  image.convertTo(gray, CV_8U, 255 * 1.0 / (max_scale - min_scale),
                  255 * (-min_scale / (max_scale - min_scale)));
  cv::applyColorMap(gray, output, ColormapToOpenCv(cm));

  return output;
}

}  // namespace replay
