#include <ceres/ceres.h>
#include <replay/flow/visualization.h>
#include <replay/multiview/composite_motion_refiner.h>
#include <replay/rendering/opengl_context.h>
#include <replay/util/image.h>
#include <Eigen/Sparse>

namespace replay {

namespace {

//// Cost functor 1:
//      Take the XY gradient of the layer1 image at the layer1 coordinates
//      (constant)
//      Take the XY gradient of the layer2 image at the layer2
//      coordinates (constant)
//      Take the layer1 flow (variable)
//      Take the layer2 flow (variable)
//      Take the unexplained residual (constant)
//      Take the initial layer1 flow (constant)
//      Take the initial layer2 flow (constant)
//
//      Subtract the initial flows from the layer flows
//      Use the subpixel flow values multiplied by the gradients to get teh
//      change in intensity. Subtract the change in intensity from the residual
//      and return it.
//
//      Return value is 3-channeled
struct DataCostFunctor {
  DataCostFunctor(const cv::Vec3d& layer1_gradient_x,
                  const cv::Vec3d& layer1_gradient_y,
                  const cv::Vec3d& layer2_gradient_x,
                  const cv::Vec3d& layer2_gradient_y,
                  const double alpha_gradient_x, const double alpha_gradient_y,
                  const cv::Vec3d& residual, const cv::Vec3d& layer2,
                  const double& alpha, const cv::Vec2d& layer1_initial_flow,
                  const cv::Vec2d& layer2_initial_flow)
      : layer1_gradient_x_(layer1_gradient_x),
        layer1_gradient_y_(layer1_gradient_y),
        layer2_gradient_x_(layer2_gradient_x),
        layer2_gradient_y_(layer2_gradient_y),
        alpha_gradient_x_(alpha_gradient_x),
        alpha_gradient_y_(alpha_gradient_y),
        residual_(residual),
        layer2_(layer2),
        alpha_(alpha),
        layer1_initial_flow_(layer1_initial_flow),
        layer2_initial_flow_(layer2_initial_flow) {}

  template <typename T>
  bool operator()(const T* const layer1_flow, const T* const layer2_flow,
                  T* e) const {
    T layer1_flow_x = layer1_flow[0] - layer1_initial_flow_[0];
    T layer1_flow_y = layer1_flow[1] - layer1_initial_flow_[1];
    T layer2_flow_x = layer2_flow[0] - layer2_initial_flow_[0];
    T layer2_flow_y = layer2_flow[1] - layer2_initial_flow_[1];
    T intensity_change[3];
    for (int c = 0; c < 3; c++) {
      intensity_change[c] = (layer1_flow_x * layer1_gradient_x_[c]) +
                            (layer1_flow_y * layer1_gradient_y_[c]) +
                            (layer2_flow_x * layer2_gradient_x_[c] * alpha_) +
                            (layer2_flow_y * layer2_gradient_y_[c] * alpha_) +
                            (layer2_[c] * alpha_gradient_x_ * layer1_flow_x) +
                            (layer2_[c] * alpha_gradient_y_ * layer1_flow_y);
      e[c] = residual_[c] - intensity_change[c];
    }

    return true;
  }

 private:
  const cv::Vec3d& layer1_gradient_x_;
  const cv::Vec3d& layer1_gradient_y_;
  const cv::Vec3d& layer2_gradient_x_;
  const cv::Vec3d& layer2_gradient_y_;
  const double alpha_gradient_x_;
  const double alpha_gradient_y_;
  const cv::Vec3d& residual_;
  const cv::Vec3d& layer2_;
  const double& alpha_;
  const cv::Vec2d& layer1_initial_flow_;
  const cv::Vec2d& layer2_initial_flow_;
};

// Cost functor 2: gradient term
//      Take two neighboring pixels (variables)
//      Take a lambda weight (constant)
//
//      Return sum of absolute values of difference between neighboring pixels
//          3 channeled return value
struct GradientCostFunctor {
  GradientCostFunctor(const double lambda, const int channels)
      : lambda_(lambda), channels_(channels) {}

  template <typename T>
  bool operator()(const T* const pixel1, const T* const pixel2, T* e) const {
    for (int c = 0; c < channels_; c++) {
      e[c] = lambda_ * (pixel2[c] - pixel1[c]);
    }
    return true;
  }

 private:
  const double lambda_;
  const int channels_;
};

}  // namespace

CompositeMotionRefiner::CompositeMotionRefiner(const int width,
                                               const int height)
    : width_(width),
      height_(height),
      coord_to_index_(cv::Size(width_, height_), -1),
      parameters_(width * height * 4) {
  // Build index remaps
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const int var_id = index_to_coord_.size();
      index_to_coord_.emplace_back(x, y);
      coord_to_index_(y, x) = var_id;
    }
  }

  // Create parameter block.
}

bool CompositeMotionRefiner::Optimize(const cv::Mat3b& layer1_img,
                                      const cv::Mat3b& layer2_img,
                                      const cv::Mat1f& alpha_img,
                                      const cv::Mat3b& composite,
                                      cv::Mat2f& layer1, cv::Mat2f& layer2,
                                      const int num_iterations) {
  cv::imshow("layer1_before", replay::FlowToColor(layer1));
  cv::imshow("layer2_before", replay::FlowToColor(layer2));

  if (layer1.rows != layer2.rows || layer1.cols != layer2.cols) {
    LOG(FATAL) << "Layer sizes must be identical";
  }

  ceres::Problem problem;

  const double flow_smoothness_lambda = 0.5;

  // For each pixel in layer 1 flow:
  //    Create one gradient cost in X direction (with SoftL1Loss)
  //    Create one gradient cost in Y direction (with SoftL1Loss)
  // For each pixel in layer 2 flow:
  //    Create one gradient cost in X direction (with SoftL1Loss)
  //    Create one gradient cost in Y direction (with SoftL1Loss)
  for (int row = 0; row < height_; row++) {
    for (int col = 0; col < width_; col++) {
      const int center_pixel = coord_to_index_(row, col) * 4;

      (const cv::Vec3d& layer1_gradient_x, const cv::Vec3d& layer1_gradient_y,
       const cv::Vec3d& layer2_gradient_x, const cv::Vec3d& layer2_gradient_y,
       const double alpha_gradient_x, const double alpha_gradient_y,
       const cv::Vec3d& residual, const cv::Vec3d& layer2, const double& alpha,
       const cv::Vec2d& layer1_initial_flow,
       const cv::Vec2d& layer2_initial_flow)

      if (col < width_ - 1 && row < height_ - 1) {
        ceres::AutoDiffCostFunction<DataCostFunctor, 2, 3>* data_cost =
            new ceres::AutoDiffCostFunction<DataCostFunctor, 2, 3>(
                new DataCostFunctor());
      }

      if (col < width_ - 1) {
        const int right_pixel = coord_to_index_(row, col + 1) * 4;

        ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>* layer1_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>(
                new GradientCostFunctor(flow_smoothness_lambda, 2));
        ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>* layer2_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>(
                new GradientCostFunctor(flow_smoothness_lambda, 2));

        problem.AddResidualBlock(layer1_cost,
                                 new ceres::SoftLOneLoss(1.0 / 255.0),
                                 parameters_.data() + center_pixel,
                                 parameters_.data() + right_pixel);
        problem.AddResidualBlock(layer2_cost,
                                 new ceres::SoftLOneLoss(1.0 / 255.0),
                                 parameters_.data() + center_pixel + 2,
                                 parameters_.data() + right_pixel + 2);
      }

      if (row < height_ - 1) {
        const int bottom_pixel = coord_to_index_(row + 1, col) * 4;

        ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>* layer1_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>(
                new GradientCostFunctor(flow_smoothness_lambda, 2));
        ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>* layer2_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>(
                new GradientCostFunctor(flow_smoothness_lambda, 2));

        problem.AddResidualBlock(layer1_cost,
                                 new ceres::SoftLOneLoss(1.0 / 255.0),
                                 parameters_.data() + center_pixel,
                                 parameters_.data() + bottom_pixel);
        problem.AddResidualBlock(layer2_cost,
                                 new ceres::SoftLOneLoss(1.0 / 255.0),
                                 parameters_.data() + center_pixel + 2,
                                 parameters_.data() + bottom_pixel + 2);
      }
    }
  }

  // Initialize the solution
  for (int row = 0; row < height_; row++) {
    for (int col = 0; col < width_; col++) {
      int pixel_index = coord_to_index_(row, col) * 4;
      for (int c = 0; c < 2; c++) {
        parameters_[pixel_index + c] = layer1(row, col)[c];
      }
      for (int c = 0; c < 2; c++) {
        parameters_[pixel_index + 2 + c] = layer2(row, col)[c];
      }
    }
  }

  return true;
}

}  // namespace replay
