#include <replay/flow/flow_from_reprojection.h>
#include <replay/rendering/image_reprojector.h>

namespace {
cv::Mat2f CreateCoordinateMatrix(int width, int height) {
  cv::Mat2f coord_mat(height, width);
  for (int row = 0; row < coord_mat.rows; row++) {
    for (int col = 0; col < coord_mat.cols; col++) {
      coord_mat(row, col) = cv::Vec2f(col / static_cast<float>(width),
                                      row / static_cast<float>(height));
    }
  }
  return coord_mat;
}
}  // namespace

namespace replay {

FlowFromReprojection::FlowFromReprojection(
    std::shared_ptr<OpenGLContext> context)
    : reprojector_(context) {}

cv::Mat2f FlowFromReprojection::Calculate(const Camera& src,
                                          const Camera& dst) {
  reprojector_.SetSourceCamera(dst);
  reprojector_.SetImage(CreateCoordinateMatrix(dst.GetImageSize().x(),
                                               dst.GetImageSize().y()));
  cv::Mat2f render;
  reprojector_.Reproject(src, &render);

  cv::Mat2f coord = CreateCoordinateMatrix(src.GetImageSize().x(),
                                           src.GetImageSize().y());

  cv::Mat2f flow(src.GetImageSize().y(), src.GetImageSize().x());
  for (int i = 0; i < flow.rows; i++) {
    for (int j = 0; j < flow.cols; j++) {
      if (render(i, j)[0] <= 0 && render(i, j)[1] <= 0) {
        flow(i, j)[0] = FLT_MAX;
        flow(i, j)[1] = FLT_MAX;
        continue;
      }
      flow(i, j)[0] = -coord(i, j)[0] * src.GetImageSize().x() +
                      render(i, j)[0] * dst.GetImageSize().x();
      flow(i, j)[1] = -coord(i, j)[1] * src.GetImageSize().y() +
                      render(i, j)[1] * dst.GetImageSize().y();
    }
  }

  return flow;
}

}  // namespace replay
