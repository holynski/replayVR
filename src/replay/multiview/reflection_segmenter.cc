#include <GCoptimization.h>
#include <replay/multiview/reflection_segmenter.h>
#include <replay/rendering/opengl_context.h>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include "opencv2/core/utility.hpp"

namespace replay {

namespace {

struct ExtraData {
  const cv::Mat3b* image;
  const cv::Mat1b* edge_mask;
};

int GradientSmoothness(int pixel1, int pixel2, int label1, int label2,
                       void* extra_data) {
  ExtraData* data = (ExtraData*)extra_data;
  const cv::Mat3b* image = data->image;
  const cv::Mat1b* edge_mask = data->edge_mask;
  if (label1 == label2) {
    return 0;
  } else {
    if ((*edge_mask)(pixel1) > 0 || (*edge_mask)(pixel2) > 0) {
      return 255 / 2 *
                       std::exp(-cv::norm((*image)(pixel1) - (*image)(pixel2)) /
                                255.0);
    }
    return 100 +
           10000 * 255 *
               std::exp(-cv::norm((*image)(pixel1) - (*image)(pixel2)) / 255.0);
  }
}

}  // namespace

ReflectionSegmenter::ReflectionSegmenter(std::shared_ptr<OpenGLContext> context,
                                         const Camera& reference_viewpoint,
                                         const cv::Mat3b& layer1_img,
                                         const cv::Mat3b& layer2_img,
                                         const Mesh& layer_1,
                                         const Mesh& layer_2)
    : context_(context),
      min_difference_(context),
      image_reprojector_(context),
      flow_calculator_(context),
      width_(reference_viewpoint.GetImageSize().x()),
      height_(reference_viewpoint.GetImageSize().y()),
      reference_viewpoint_(reference_viewpoint),
      layer1_img_(layer1_img),
      layer2_img_(layer2_img),
      layer_1_mesh_id_(context_->UploadMesh(layer_1)),
      layer_2_mesh_id_(context_->UploadMesh(layer_2)),
      reflection_cost_(height_, width_, 0.0),
      diffuse_cost_(height_, width_, 0.0),
      cost_count_(height_, width_, 1.0) {
  if (layer1_img_.rows != layer2_img_.rows ||
      layer1_img_.cols != layer2_img_.cols) {
    LOG(FATAL) << "Layer sizes must be identical";
  }
}

bool ReflectionSegmenter::AddImage(const cv::Mat3b& image,
                                   const Camera& camera) {
  context_->BindMesh(layer_1_mesh_id_);
  cv::Mat2f layer1_flow =
      flow_calculator_.Calculate(camera, reference_viewpoint_);
  cv::Mat3b first_layer;
  image_reprojector_.SetImage(layer1_img_);
  image_reprojector_.SetSourceCamera(reference_viewpoint_);
  image_reprojector_.Reproject(camera, &first_layer, 0.25);

  context_->BindMesh(layer_2_mesh_id_);
  cv::Mat3b second_layer;
  image_reprojector_.SetImage(layer2_img_);
  image_reprojector_.SetSourceCamera(reference_viewpoint_);
  image_reprojector_.Reproject(camera, &second_layer, 0.25);

  const int rows = image.rows;
  const int cols = image.cols;

  cv::Mat3b residual_img =
      min_difference_.GetDifference(image, first_layer, 10);
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      const cv::Point_<float> base_coord(col, row);
      cv::Vec2f flow_layer1 = layer1_flow(row, col);
      const cv::Point_<int> layer1_coord(std::round((col + flow_layer1[0])),
                                         std::round((row + flow_layer1[1])));

      if (layer1_coord.x < 0 || layer1_coord.y < 0 ||
          layer1_coord.x >= width_ || layer1_coord.y >= height_) {
        continue;
      }

      cv::Vec3f layer1;
      cv::Vec3f layer2;
      cv::Vec3f observation;
      cv::Vec3f residual;
      cv::Vec3f composite;

      for (int c = 0; c < 3; c++) {
        layer1[c] = first_layer(row, col)[c];
        layer2[c] = second_layer(row, col)[c];
        observation[c] = image(row, col)[c];
        residual[c] = residual_img(row, col)[c];
      }

      if (layer2 == cv::Vec3f(0, 0, 0) || layer1 == cv::Vec3f(0, 0, 0)) {
        continue;
      }

      for (int c = 0; c < 3; c++) {
        composite[c] = std::fmin(
            static_cast<int>(layer1[c]) + static_cast<int>(layer2[c]), 255);
      }

      cv::Vec3f reflection_error = layer2 - residual;
      cv::Vec3f diffuse_error = residual;

      for (int c = 0; c < 3; c++) {
        reflection_error[c] = std::fmax(0, reflection_error[c]);
        diffuse_error[c] = std::fmax(0, diffuse_error[c]);
      }

      reflection_cost_(layer1_coord.y, layer1_coord.x) +=
          cv::norm(reflection_error);
      // diffuse_cost_(layer1_coord.y, layer1_coord.x) +=
      // cv::norm(diffuse_error);
      diffuse_cost_(layer1_coord.y, layer1_coord.x) =
          std::fmax(cv::norm(diffuse_error),
                    diffuse_cost_(layer1_coord.y, layer1_coord.x));
      cost_count_(layer1_coord.y, layer1_coord.x) += 1;
    }
  }

  return true;
}

bool ReflectionSegmenter::Optimize(cv::Mat1b& mask,
                                   const cv::Mat1b& candidate_edges) {
  cv::Mat1d normalized_reflection_cost = reflection_cost_ / cost_count_;
  cv::Mat1d normalized_diffuse_cost = diffuse_cost_;

  cv::imshow("reflection", normalized_reflection_cost / 255.0);
  cv::imshow("diffuse", normalized_diffuse_cost / 255.0);
  cv::imwrite("/Users/holynski/reflection.png", normalized_reflection_cost);
  cv::imwrite("/Users/holynski/diffuse.png", normalized_diffuse_cost);
  cv::waitKey(1);

  // cv::Mat1b edge_mask = cv::Mat1b::ones(normalized_reflection_cost.size());

  // cv::Ptr<cv::line_descriptor::LSDDetector> bd =
  // cv::line_descriptor::LSDDetector::createLSDDetector();
  // std::vector<cv::line_descriptor::KeyLine> lines;
  // bd->detect(layer1_img_, lines, 1, 1, cv::Mat());
  // for (size_t i = 0; i < lines.size(); i++) {
  // cv::line_descriptor::KeyLine kl = lines[i];
  // if (kl.octave == 0) {
  //[> get extremes of line <]
  // cv::Point pt1 = cv::Point(kl.startPointX, kl.startPointY);
  // cv::Point pt2 = cv::Point(kl.endPointX, kl.endPointY);

  //[> draw line <]
  // line(edge_mask, pt1, pt2, cv::Scalar(255), 1);
  //}
  //}

  GCoptimizationGridGraph* mrf =
      new GCoptimizationGridGraph(width_, height_, 2);

  for (int pixel = 0; pixel < width_ * height_; pixel++) {
    mrf->setDataCost(pixel, 0, diffuse_cost_(pixel));
    mrf->setDataCost(pixel, 1, reflection_cost_(pixel) / cost_count_(pixel));
  }

  ExtraData data;
  data.image = &layer1_img_;
  data.edge_mask = &candidate_edges;

  mrf->setSmoothCost(&GradientSmoothness, reinterpret_cast<void*>(&data));

  mask = cv::Mat1b(height_, width_);

  try {
    LOG(INFO) << "Before optimization energy is " << mrf->compute_energy();

    LOG(INFO) << "Running alpha expansion...";
    mrf->expansion(2);  // run expansion for 2 iterations. For
                        // swap use gc->swap(num_iterations);
    LOG(INFO) << "After optimization energy is " << mrf->compute_energy();

    for (int pixel = 0; pixel < width_ * height_; pixel++) {
      const int label = mrf->whatLabel(pixel);
      mask(pixel) = label * 255;
    }

    delete mrf;
  } catch (GCException e) {
    e.Report();
    return false;
  }

  return true;
}

}  // namespace replay
