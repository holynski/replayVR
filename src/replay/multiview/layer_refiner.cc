#include <ceres/ceres.h>
#include <replay/multiview/layer_refiner.h>
#include <replay/rendering/opengl_context.h>

namespace replay {

namespace {

struct CostFunctor {
  CostFunctor(const cv::Vec3b& observation) : observation_(observation) {}
  template <typename T>
  bool operator()(T const* const* parameters, T* residual) const {
    T const* layer1 = parameters[0];
    T const* layer2 = parameters[1];
    T const* alpha = parameters[2];
    for (int c = 0; c < 3; c++) {
      residual[c] =
          ceres::abs((layer1[c] * alpha[0] + (T(1.0) - alpha[0]) * layer2[c]) -
                     T(observation_[c]));
    }
    return true;
  }

  cv::Vec3b observation_;
};

}  // namespace

LayerRefiner::LayerRefiner(const int width, const int height)
    : width_(width), height_(height) {
  parameter_blocks_.resize(width_ * height_ * 7, 0.0);
}

bool LayerRefiner::AddImage(const cv::Mat3b& image,
                            const cv::Mat2f& layer1_mapping,
                            const cv::Mat2f& layer2_mapping) {
  const int rows = image.rows;
  const int cols = image.cols;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      const cv::Vec3b observation = image(row, col);

      const cv::Point_<int> base_coord(col, row);
      cv::Vec2f flow_layer1 = layer1_mapping(row, col);
      cv::Vec2f flow_layer2 = layer2_mapping(row, col);
      const cv::Point layer1_coord =
          base_coord + cv::Point_<int>(std::round(flow_layer1[0]),
                                       std::round(flow_layer1[1]));
      const cv::Point layer2_coord =
          base_coord + cv::Point_<int>(std::round(flow_layer2[0]),
                                       std::round(flow_layer2[1]));

      if (layer1_coord.x < 0 || layer1_coord.y < 0 || layer2_coord.x < 0 ||
          layer2_coord.y < 0 || layer1_coord.x >= width_ ||
          layer1_coord.y >= height_ || layer2_coord.x >= width_ ||
          layer2_coord.y >= height_) {
        continue;
      }

      int parameter_index_layer1 =
          (layer1_coord.y * width_ + layer1_coord.x) * 7;
      int parameter_index_layer2 =
          3 + (layer2_coord.y * width_ + layer2_coord.x) * 7;
      int parameter_index_alpha =
          6 + (layer1_coord.y * width_ + layer1_coord.x) * 7;

      ceres::DynamicAutoDiffCostFunction<CostFunctor>* costfn =
          new ceres::DynamicAutoDiffCostFunction<CostFunctor>(
              new CostFunctor(observation));

      // Layer 1 intensities
      costfn->AddParameterBlock(3);

      // Layer 2 intensities
      costfn->AddParameterBlock(3);

      // Add one parameter to model the alphas
      costfn->AddParameterBlock(1);

      costfn->SetNumResiduals(3);

      problem_.AddResidualBlock(
          costfn, NULL, parameter_blocks_.data() + parameter_index_layer1,
          parameter_blocks_.data() + parameter_index_layer2,
          parameter_blocks_.data() + parameter_index_alpha);
      for (int c = 0; c < 3; c++) {
        problem_.SetParameterLowerBound(
            parameter_blocks_.data() + parameter_index_layer1, c, 0);
        problem_.SetParameterUpperBound(
            parameter_blocks_.data() + parameter_index_layer1, c, 255);
      }
      for (int c = 0; c < 3; c++) {
        problem_.SetParameterLowerBound(
            parameter_blocks_.data() + parameter_index_layer1, c, 0);
        problem_.SetParameterUpperBound(
            parameter_blocks_.data() + parameter_index_layer1, c, 255);
      }
      problem_.SetParameterLowerBound(
          parameter_blocks_.data() + parameter_index_alpha, 0, 0);
      problem_.SetParameterUpperBound(
          parameter_blocks_.data() + parameter_index_alpha, 0, 1);
    }
  }

  return true;
}

bool LayerRefiner::Optimize(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img,
                            cv::Mat1f& alpha_mask) {
  if (layer1_img.rows != layer2_img.rows ||
      layer1_img.cols != layer2_img.cols) {
    LOG(ERROR) << "Layer sizes must be identical";
    return false;
  }

  height_ = layer1_img.rows;
  width_ = layer1_img.cols;
  LOG(ERROR) << "Creating parameter block";

  // Convert the initialization to doubles in 1D arrays

  for (int y = 0; y < height_; y++) {
    for (int x = 0; x < width_; x++) {
      int parameter_index = (y * width_ + x) * 7;
      for (int c = 0; c < 3; c++) {
        parameter_blocks_[parameter_index + c] =
            static_cast<double>(layer1_img(y, x)[c]);
      }
      for (int c = 0; c < 3; c++) {
        parameter_blocks_[parameter_index + 3 + c] =
            static_cast<double>(layer2_img(y, x)[c]);
      }
      parameter_blocks_[parameter_index + 6] =
          static_cast<double>(alpha_mask(y, x));
    }
  }

  LOG(ERROR) << "Calling Ceres::Solve";
  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.max_num_iterations = 50;
  options.minimizer_progress_to_stdout = true;
  // options.function_tolerance = 1e-8;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_, &summary);
  LOG(INFO) << summary.FullReport();

  for (int y = 0; y < height_; y++) {
    for (int x = 0; x < width_; x++) {
      int parameter_index = (y * width_ + x) * 7;
      for (int c = 0; c < 3; c++) {
        layer1_img(y, x)[c] = parameter_blocks_[parameter_index + c];
      }
      for (int c = 0; c < 3; c++) {
        layer2_img(y, x)[c] = parameter_blocks_[parameter_index + 3 + c];
      }
      alpha_mask(y, x) = parameter_blocks_[parameter_index + 6];
    }
  }

  return true;
}

}  // namespace replay
