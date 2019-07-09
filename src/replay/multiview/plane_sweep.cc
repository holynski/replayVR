#include "replay/multiview/plane_sweep.h"
#include "replay/image/image_stack_analyzer.h"
#include "replay/io/read_float_image.h"
#include "replay/io/write_float_image.h"
#include "replay/util/filesystem.h"
#include "replay/util/progress_bar.h"
#include "replay/util/timer.h"

namespace replay {

PlaneSweep::PlaneSweep(std::shared_ptr<OpenGLContext> context,
                       const std::string& cache_directory)
    : context_(context),
      sad_(context_),
      reprojector_(context_),
      cache_directory_(cache_directory) {
  CHECK(context->IsInitialized()) << "Initialize renderer first!";
}

void PlaneSweep::AddView(const Camera& camera, const cv::Mat3b& image,
                         const cv::Mat1b& valid_pixels) {
  cameras_.push_back(&camera);
  images_.push_back(image);
  masks_.push_back(valid_pixels);
}

PlaneSweepResult PlaneSweep::Sweep(const Camera& viewpoint,
                                   const float min_depth, const float max_depth,
                                   const int num_steps) {
  float step_size = (max_depth - min_depth) / num_steps;
  const Eigen::Vector3f plane_normal =
      viewpoint.GetLookAt().cast<float>().normalized();
  std::unordered_map<float, cv::Mat1f> cost_volume;

  int step_id = 0;
  PlaneSweepResult result;
  for (float depth = min_depth; depth <= max_depth; depth += step_size) {
    step_id++;

    const float plane_d =
        (viewpoint.GetPosition().cast<float>() + plane_normal * depth)
            .dot(plane_normal);
    Eigen::Vector4f plane(plane_normal[0], plane_normal[1], plane_normal[2],
                          plane_d);

    Mesh plane_mesh = Mesh::Plane(plane.cast<float>().head<3>() * plane[3],
                                  plane.cast<float>().head<3>(),
                                  Eigen::Vector2f(50000, 50000));
    const int plane_id = context_->UploadMesh(plane_mesh);
    context_->BindMesh(plane_id);

    context_->BindMesh(plane_id);
    result.cost_volume[depth] = ReadFloatImage(
        JoinPath(cache_directory_, "cost_" + std::to_string(depth) + ".bin"));
    result.num_samples[depth] = ReadFloatImage(
        JoinPath(cache_directory_, "count_" + std::to_string(depth) + ".bin"));
    result.mean_images[depth] = cv::imread(
        JoinPath(cache_directory_, "mean_" + std::to_string(depth) + ".png"));

    PrintProgress(
        1000 * (depth - min_depth), 1000 * (max_depth - min_depth), "Plane sweep",
        (result.cost_volume[depth].empty() ? "Loaded." : "Computing..."));

    result.depths.push_back(depth);
    result.mesh_ids[depth] = plane_id;
    result.meshes[depth] = plane_mesh;
    if (result.cost_volume[depth].empty()) {
      SimpleTimer timer;
      ImageStackAnalyzer::Options options;
      options.compute_max = false;
      options.compute_min = false;
      options.compute_median = false;
      ImageStackAnalyzer layer_statistics(options);
      for (int cam = 0; cam < cameras_.size(); cam++) {
        cv::Mat3b reprojected;
        CHECK(reprojector_.SetSourceCamera(*cameras_[cam]));
        if (!masks_[cam].empty()) {
          images_[cam].setTo(cv::Vec3b(0, 0, 0), masks_[cam] == 0);
        }
        CHECK(reprojector_.SetImage(images_[cam]));
        CHECK(reprojector_.Reproject(viewpoint, &reprojected));
        cv::Mat1b mask;
        cv::cvtColor(reprojected, mask, cv::COLOR_BGR2GRAY);
        mask = mask > 0;
        layer_statistics.AddImage(reprojected, mask);
      }
      cv::Mat3f variance_rgb = layer_statistics.GetVariance();
      cv::Mat1f variance_gray;
      cv::cvtColor(variance_rgb, variance_gray, cv::COLOR_BGR2GRAY);

      result.cost_volume[depth] = variance_gray;
      result.mean_images[depth] = layer_statistics.GetMean();
      result.num_samples[depth] = layer_statistics.GetCount();

      if (!cache_directory_.empty()) {
        replay::WriteFloatImage(
            replay::JoinPath(cache_directory_,
                             "count_" + std::to_string(depth) + ".bin"),
            result.num_samples[depth]);
        replay::WriteFloatImage(
            replay::JoinPath(cache_directory_,
                             "cost_" + std::to_string(depth) + ".bin"),
            variance_gray);
        cv::imwrite(replay::JoinPath(cache_directory_,
                                     "mean_" + std::to_string(depth) + ".png"),
                    result.mean_images[depth]);
      }
    }
    // cv::GaussianBlur(result.cost_volume[depth], result.cost_volume[depth],
    // cv::Size(11,11), 6);
  }
  return result;
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
