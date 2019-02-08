#include "replay/multiview/plane_sweep.h"

namespace replay {

PlaneSweep::PlaneSweep(std::shared_ptr<OpenGLContext> context)
    : context_(context), sad_(context_), reprojector_(context_) {
  CHECK(context->IsInitialized()) << "Initialize renderer first!";
}

cv::Mat1f PlaneSweep::GetCost(const Camera& camera1, const Camera& camera2,
                              const cv::Mat3b& image1,
                              const cv::Mat1b& valid_pixels1,
                              const cv::Mat3b& image2,
                              const cv::Mat1b& valid_pixels2,
                              const Eigen::Vector4f& plane) {
  const Eigen::Vector2i& image_size = camera1.GetImageSize();
  cv::Mat3b reprojected(image_size.x(), image_size.y());
  cv::Mat reprojected_mask(image_size.x(), image_size.y(), CV_8UC3);

  Mesh plane_mesh = Mesh::Plane(plane.head<3>() * plane[3], plane.head<3>(),
                                Eigen::Vector2f(5000, 5000));
  const int plane_mesh_id = context_->UploadMesh(plane_mesh);
  context_->BindMesh(plane_mesh_id);

  CHECK(reprojector_.SetSourceCamera(camera2));
  CHECK(reprojector_.SetImage(image2));
  CHECK(reprojector_.Reproject(camera1, &reprojected));
  if (!valid_pixels2.empty()) {
    cv::Mat3b rgb;
    cv::cvtColor(valid_pixels2, rgb, cv::COLOR_GRAY2RGB);
    CHECK(reprojector_.SetImage(valid_pixels2));
    CHECK(reprojector_.Reproject(camera1, &reprojected_mask));
    cv::cvtColor(reprojected_mask, reprojected_mask, cv::COLOR_RGB2GRAY);
  }

  cv::Mat1f cost = sad_.GetDifference(
      image1, reprojected, valid_pixels1,
      valid_pixels2.empty() ? valid_pixels2 : reprojected_mask > 0);
  if (!valid_pixels1.empty()) {
    cost.setTo(0, valid_pixels1 == 0);
  }
  if (!valid_pixels2.empty()) {
    cost.setTo(0, reprojected_mask == 0);
  }
  return cost;
}

std::unordered_map<float, cv::Mat1f> PlaneSweep::Sweep(
    const Camera& camera1, const Camera& camera2, const cv::Mat3b& image1,
    const cv::Mat1b& valid_pixels1, const cv::Mat3b& image2,
    const cv::Mat1b& valid_pixels2, const float min_depth,
    const float max_depth, const int num_steps) {
  float step_size = (max_depth - min_depth) / num_steps;
  const Eigen::Vector3f plane_normal =
      camera1.GetLookAt().cast<float>().normalized();
  std::unordered_map<float, cv::Mat1f> cost_volume;
  for (float depth = min_depth; depth <= max_depth; depth += step_size) {
    const float plane_d =
        (camera1.GetPosition().cast<float>() + plane_normal * depth)
            .dot(plane_normal);
    Eigen::Vector4f plane(plane_normal[0], plane_normal[1], plane_normal[2],
                          plane_d);
    cost_volume[depth] = GetCost(camera1, camera2, image1, valid_pixels1,
                                 image2, valid_pixels2, plane);
  }
  return cost_volume;
}

}  // namespace replay
