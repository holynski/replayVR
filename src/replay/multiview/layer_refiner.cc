#include <ceres/ceres.h>
#include <replay/flow/optical_flow.h>
#include <replay/flow/optical_flow_aligner.h>
#include <replay/flow/visualization.h>
#include <replay/multiview/layer_refiner.h>
#include <replay/rendering/opengl_context.h>
#include <Eigen/Sparse>

namespace replay {

namespace {

// Cost functor 1: Data term
//      Take observed pixel (constant)
//      Take layer1 pixel (variable)
//      Take layer2 pixel (variable)
//      Take alpha pixel (variable)
//
//      Return difference (observed - (layer1 + alpha*layer2))
//        3-channeled return value
struct DataCostFunctor {
  DataCostFunctor(const cv::Vec3d& observed_color, const int channels)
      : observed_color_(observed_color), channels_(channels) {}

  template <typename T>
  bool operator()(const T* const layer1, const T* const layer2,
                  const T* const alpha, T* e) const {
    for (int c = 0; c < channels_; c++) {
      e[c] = T(observed_color_[c]) - (layer1[c] + alpha[0] * layer2[c]);
    }
    return true;
  }

 private:
  const cv::Vec3d observed_color_;
  const int channels_;
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

// Cost functor 3: Correlation term
//      Take two neighboring pixels in layer 1 (variables)
//      Take two neighboring pixels in layer 2 (variables)
//      Take a lambda weight (constant)
//
//      Return squared_norm(layer1_difference) * squared_norm(layer2_difference)
struct CorrelationCostFunctor {
  CorrelationCostFunctor(const double lambda) : lambda_(lambda) {}

  template <typename T>
  bool operator()(const T* const layer1_pixel1, const T* const layer1_pixel2,
                  const T* const layer2_pixel1, const T* const layer2_pixel2,
                  T* e) const {
    T sq_norm1 = T(0);
    T sq_norm2 = T(0);
    for (int c = 0; c < 3; c++) {
      sq_norm1 += ceres::pow(layer1_pixel2[c] - layer1_pixel1[c], 2);
      sq_norm2 += ceres::pow(layer2_pixel2[c] - layer2_pixel1[c], 2);
    }
    e[0] = sq_norm1 * sq_norm2 * lambda_;
    return true;
  }

 private:
  const double lambda_;
};

}  // namespace

LayerRefiner::LayerRefiner(const int width, const int height)
    : width_(width),
      height_(height),
      coord_to_index_(cv::Size(width_, height_), -1),
      num_images_(0) {
  parameters_.resize(width_ * height_ * 7);

  // Build index remaps
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const int var_id = index_to_coord_.size();
      index_to_coord_.emplace_back(x, y);
      coord_to_index_(y, x) = var_id;
    }
  }
}

bool LayerRefiner::AddImage(const cv::Mat3b& image,
                            const cv::Mat2f& flow_to_layer1,
                            const cv::Mat2f& flow_to_layer2,
                            const cv::Mat1b& valid_pixels) {
  // Each image that is added corresponds to:
  //    For each pixel:
  //        Create one data cost term
  cv::Mat3b fg_splat(cv::Size(width_, height_), cv::Vec3b(255, 255, 255));
  cv::Mat3b bg_splat(cv::Size(width_, height_), cv::Vec3b(255, 255, 255));

  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      if (!valid_pixels.empty()) {
        if (valid_pixels(row, col) == 0) {
          continue;
        }
      }
      const cv::Vec2f coord(col, row);
      const cv::Vec2f layer1_flow = flow_to_layer1(row, col);
      const cv::Vec2f layer2_flow = flow_to_layer2(row, col);
      const cv::Vec2f flowed_layer1 = coord + layer1_flow;
      const cv::Vec2f flowed_layer2 = coord + layer2_flow;

      const cv::Vec2i flowed_layer1_pixel(std::round(flowed_layer1[0]),
                                          std::round(flowed_layer1[1]));
      const cv::Vec2i flowed_layer2_pixel(std::round(flowed_layer2[0]),
                                          std::round(flowed_layer2[1]));

      if (flowed_layer1_pixel[0] < 0 || flowed_layer1_pixel[1] < 0 ||
          flowed_layer2_pixel[0] < 0 || flowed_layer2_pixel[1] < 0 ||
          flowed_layer1_pixel[0] >= width_ ||
          flowed_layer1_pixel[1] >= height_ ||
          flowed_layer2_pixel[0] >= width_ ||
          flowed_layer2_pixel[1] >= height_) {
        continue;
      }

      const int layer1_index =
          coord_to_index_(flowed_layer1_pixel[1], flowed_layer1_pixel[0]) * 7;
      const int alpha_index =
          coord_to_index_(flowed_layer1_pixel[1], flowed_layer1_pixel[0]) * 7 +
          6;
      const int layer2_index =
          coord_to_index_(flowed_layer2_pixel[1], flowed_layer2_pixel[0]) * 7 +
          3;

      const cv::Vec3b observed_color = image(row, col);
      fg_splat(flowed_layer1_pixel[1], flowed_layer1_pixel[0]) = observed_color;
      bg_splat(flowed_layer2_pixel[1], flowed_layer2_pixel[0]) = observed_color;
      cv::Vec3d observed_color_scaled;

      for (int c = 0; c < 3; c++) {
        observed_color_scaled[c] = observed_color[c] / 255.0;
      }

      ceres::AutoDiffCostFunction<DataCostFunctor, 3, 3, 3, 1>* costfn =
          new ceres::AutoDiffCostFunction<DataCostFunctor, 3, 3, 3, 1>(
              new DataCostFunctor(observed_color_scaled, 3));

      problem_.AddResidualBlock(costfn, new ceres::SoftLOneLoss(1.0 / 255.0),
                                parameters_.data() + layer1_index,
                                parameters_.data() + layer2_index,
                                parameters_.data() + alpha_index);
    }
  }
  num_images_++;
  return true;
}

bool LayerRefiner::Optimize(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img,
                            cv::Mat1f& alpha, const int num_iterations) {
  CHECK_GT(num_images_, 0);

  // Here  we use the weights described in Xue et al. "A Computational Approach
  // for Obstruction-Free Photography". The weights provided in the paper are
  // tuned for a fixed number of images (5), and the number of data cost terms
  // increase with each added image (but the smoothness costs do not), so we
  // need to scale the smoothness weights by the number of images.
  const double layer_smoothness_lambda = 0.1;// * (num_images_ / 5.0);
  const double alpha_smoothness_lambda = 1.0;// * (num_images_ / 5.0);
  const double correlation_lambda = 3000;// * (num_images_ / 5.0);

  // For each pixel in layer 1:
  //    Create one gradient cost in X direction (with SoftL1Loss)
  //    Create one gradient cost in Y direction (with SoftL1Loss)
  // For each pixel in layer 2:
  //    Create one gradient cost in X direction (with SoftL1Loss)
  //    Create one gradient cost in Y direction (with SoftL1Loss)
  // For each pixel in alpha:
  //    Create one gradient cost in X direction (with TrivialLoss)
  //    Create one gradient cost in Y direction (with TrivialLoss)
  // For each pixel in layer1/layer2:
  //    Create one correlation cost in X direction (with TrivialLoss)
  //    Create one correlation cost in Y direction (with TrivialLoss)

  for (int row = 0; row < height_; row++) {
    for (int col = 0; col < width_; col++) {
      const int center_pixel = coord_to_index_(row, col) * 7;
      if (col < width_ - 1) {
        const int right_pixel = coord_to_index_(row, col + 1) * 7;

        ceres::AutoDiffCostFunction<GradientCostFunctor, 3, 3, 3>* layer1_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 3, 3, 3>(
                new GradientCostFunctor(layer_smoothness_lambda, 3));
        ceres::AutoDiffCostFunction<GradientCostFunctor, 3, 3, 3>* layer2_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 3, 3, 3>(
                new GradientCostFunctor(layer_smoothness_lambda, 3));
        ceres::AutoDiffCostFunction<GradientCostFunctor, 1, 1, 1>* alpha_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 1, 1, 1>(
                new GradientCostFunctor(alpha_smoothness_lambda, 1));
        ceres::AutoDiffCostFunction<CorrelationCostFunctor, 1, 3, 3, 3, 3>*
            correlation_cost =
                new ceres::AutoDiffCostFunction<CorrelationCostFunctor, 1, 3, 3,
                                                3, 3>(
                    new CorrelationCostFunctor(correlation_lambda));

        problem_.AddResidualBlock(layer1_cost,
                                  new ceres::SoftLOneLoss(1.0 / 255.0),
                                  parameters_.data() + center_pixel,
                                  parameters_.data() + right_pixel);
        problem_.AddResidualBlock(layer2_cost,
                                  new ceres::SoftLOneLoss(1.0 / 255.0),
                                  parameters_.data() + center_pixel + 3,
                                  parameters_.data() + right_pixel + 3);
        problem_.AddResidualBlock(alpha_cost, new ceres::SoftLOneLoss(1.0 / 255.0),
                                  parameters_.data() + center_pixel + 6,
                                  parameters_.data() + right_pixel + 6);
        problem_.AddResidualBlock(correlation_cost, new ceres::TrivialLoss(),
                                  parameters_.data() + center_pixel,
                                  parameters_.data() + right_pixel,
                                  parameters_.data() + center_pixel + 3,
                                  parameters_.data() + right_pixel + 3);
      }

      if (row < height_ - 1) {
        const int bottom_pixel = coord_to_index_(row + 1, col) * 7;

        ceres::AutoDiffCostFunction<GradientCostFunctor, 3, 3, 3>* layer1_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 3, 3, 3>(
                new GradientCostFunctor(layer_smoothness_lambda, 3));
        ceres::AutoDiffCostFunction<GradientCostFunctor, 3, 3, 3>* layer2_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 3, 3, 3>(
                new GradientCostFunctor(layer_smoothness_lambda, 3));
        ceres::AutoDiffCostFunction<GradientCostFunctor, 1, 1, 1>* alpha_cost =
            new ceres::AutoDiffCostFunction<GradientCostFunctor, 1, 1, 1>(
                new GradientCostFunctor(alpha_smoothness_lambda, 1));
        ceres::AutoDiffCostFunction<CorrelationCostFunctor, 1, 3, 3, 3, 3>*
            correlation_cost =
                new ceres::AutoDiffCostFunction<CorrelationCostFunctor, 1, 3, 3,
                                                3, 3>(
                    new CorrelationCostFunctor(correlation_lambda));

        problem_.AddResidualBlock(layer1_cost,
                                  new ceres::SoftLOneLoss(1.0 / 255.0),
                                  parameters_.data() + center_pixel,
                                  parameters_.data() + bottom_pixel);
        problem_.AddResidualBlock(layer2_cost,
                                  new ceres::SoftLOneLoss(1.0 / 255.0),
                                  parameters_.data() + center_pixel + 3,
                                  parameters_.data() + bottom_pixel + 3);
        problem_.AddResidualBlock(alpha_cost, new ceres::SoftLOneLoss(1.0 / 255.0),
                                  parameters_.data() + center_pixel + 6,
                                  parameters_.data() + bottom_pixel + 6);
        problem_.AddResidualBlock(correlation_cost, new ceres::TrivialLoss(),
                                  parameters_.data() + center_pixel,
                                  parameters_.data() + bottom_pixel,
                                  parameters_.data() + center_pixel + 3,
                                  parameters_.data() + bottom_pixel + 3);
      }
    }
  }

  // Initialize the solution
  for (int row = 0; row < height_; row++) {
    for (int col = 0; col < width_; col++) {
      int pixel_index = coord_to_index_(row, col) * 7;
      for (int c = 0; c < 3; c++) {
        parameters_[pixel_index + c] = layer1_img(row, col)[c] / 255.0;
      }
      for (int c = 0; c < 3; c++) {
        parameters_[pixel_index + 3 + c] = layer2_img(row, col)[c] / 255.0;
      }
      parameters_[pixel_index + 6] = 1.0 * alpha(row, col);
    }
  }

  // Set the bounds:
  //    Alpha = 0...1
  //    Layer1 = 0...1
  //    Layer2 = 0...1
  for (int row = 0; row < height_; row++) {
    for (int col = 0; col < width_; col++) {
      int pixel_index = coord_to_index_(row, col) * 7;
      for (int c = 0; c < 3; c++) {
        problem_.SetParameterLowerBound(parameters_.data() + pixel_index, c, 0);
        problem_.SetParameterUpperBound(parameters_.data() + pixel_index, c, 1);
      }
      problem_.SetParameterLowerBound(parameters_.data() + 6 + pixel_index, 0,
                                      0);
      problem_.SetParameterUpperBound(parameters_.data() + 6 + pixel_index, 0,
                                      1);
      for (int c = 0; c < 3; c++) {
        problem_.SetParameterLowerBound(parameters_.data() + 3 + pixel_index, c,
                                        0);
        problem_.SetParameterUpperBound(parameters_.data() + 3 + pixel_index, c,
                                        1);
      }
    }
  }

  //    Optimize!
  LOG(INFO) << "Solving with Ceres...";
  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::CGNR;
  options.preconditioner_type = ceres::JACOBI;
  options.max_num_iterations = 200;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = 1e-8;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_, &summary);
  LOG(INFO) << summary.FullReport();

  LOG(INFO) << "Copying solution...";
  for (int row = 0; row < height_; row++) {
    for (int col = 0; col < width_; col++) {
      int pixel_index = coord_to_index_(row, col) * 7;
      for (int c = 0; c < 3; c++) {
        layer1_img(row, col)[c] = parameters_[pixel_index + c] * 255.0;
      }
      for (int c = 0; c < 3; c++) {
        layer2_img(row, col)[c] = parameters_[pixel_index + 3 + c] * 255.0;
      }
      alpha(row, col) = parameters_[pixel_index + 6];
    }
  }

  return true;
}

}  // namespace replay
