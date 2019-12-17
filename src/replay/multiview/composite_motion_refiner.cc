#include <ceres/ceres.h>
#include <replay/flow/visualization.h>
#include <replay/multiview/composite_motion_refiner.h>
#include <replay/rendering/opengl_context.h>
#include <replay/util/image.h>
#include <Eigen/Sparse>
#include "ceres/ceres.h"
#include "jet_extras.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace replay {

namespace {

template <typename TPixel>
void LinearInitAxis(TPixel x, int size, int* x1, int* x2, TPixel* dx) {
  const int ix = static_cast<int>(x);
  if (ix < 0) {
    *x1 = 0;
    *x2 = 0;
    *dx = 1.0;
  } else if (ix > size - 2) {
    *x1 = size - 1;
    *x2 = size - 1;
    *dx = 1.0;
  } else {
    *x1 = ix;
    *x2 = ix + 1;
    *dx = *x2 - x;
  }
}

/// Linear interpolation.
template <typename T, class TImage>
void SampleLinear(const TImage& intensityImage,
                  const TImage& intensityGradientX,
                  const TImage& intensityGradientY,
                  typename TImage::value_type y, typename TImage::value_type x,
                  T* sample) {
  typedef TImage ImageType;
  typedef typename ImageType::value_type PixelType;

  int x1, y1, x2, y2;
  PixelType dx, dy;

  // Take the upper left corner as integer pixel positions.
  x -= T(0.5);
  y -= T(0.5);

  LinearInitAxis(y, intensityImage.rows, &y1, &y2, &dy);
  LinearInitAxis(x, intensityImage.cols, &x1, &x2, &dx);

  // Sample intensity
  const T im11 = T(intensityImage(y1, x1));
  const T im12 = T(intensityImage(y1, x2));
  const T im21 = T(intensityImage(y2, x1));
  const T im22 = T(intensityImage(y2, x2));

  sample[0] = (dy * (dx * im11 + (1.0 - dx) * im12) +
               (1 - dy) * (dx * im21 + (1.0 - dx) * im22));

  // Sample gradient x
  const T gradx11 = T(intensityGradientX(y1, x1));
  const T gradx12 = T(intensityGradientX(y1, x2));
  const T gradx21 = T(intensityGradientX(y2, x1));
  const T gradx22 = T(intensityGradientX(y2, x2));

  sample[1] = (dy * (dx * gradx11 + (1.0 - dx) * gradx12) +
               (1 - dy) * (dx * gradx21 + (1.0 - dx) * gradx22));

  // Sample gradient y
  const T grady11 = T(intensityGradientY(y1, x1));
  const T grady12 = T(intensityGradientY(y1, x2));
  const T grady21 = T(intensityGradientY(y2, x1));
  const T grady22 = T(intensityGradientY(y2, x2));

  sample[2] = (dy * (dx * grady11 + (1.0 - dx) * grady12) +
               (1 - dy) * (dx * grady21 + (1.0 - dx) * grady22));
}

// Sample the image at position (x, y) but use the gradient to
// propagate derivatives from x and y. This is needed to integrate the numeric
// image gradients with Ceres's autodiff framework.
template <typename T, class TImage>
T SampleWithDerivative(const TImage& intensityImage,
                       const TImage& intensityGradientX,
                       const TImage& intensityGradientY, const T& x,
                       const T& y) {
  typedef TImage ImageType;
  typedef typename ImageType::value_type PixelType;

  PixelType scalar_x = ceres::JetOps<T>::GetScalar(x);
  PixelType scalar_y = ceres::JetOps<T>::GetScalar(y);

  PixelType sample[3];
  // Sample intensity image and gradients
  SampleLinear(intensityImage, intensityGradientX, intensityGradientY, scalar_y,
               scalar_x, sample);
  T xy[2] = {x, y};
  return ceres::Chain<PixelType, 2, T>::Rule(sample[0], sample + 1, xy);
}

//// Cost functor 1:
//
//      Subtract the initial flows from the layer flows
//      Use the subpixel flow values multiplied by the gradients to get teh
//      change in intensity. Subtract the change in intensity from the residual
//      and return it.
//
//      Return value is 3-channeled
struct DataCostFunctor {
  DataCostFunctor(const std::vector<cv::Mat1d>& layer1_image,
                  const std::vector<cv::Mat1d>& layer1_gradient_x,
                  const std::vector<cv::Mat1d>& layer1_gradient_y,
                  const std::vector<cv::Mat1d>& layer2_image,
                  const std::vector<cv::Mat1d>& layer2_gradient_x,
                  const std::vector<cv::Mat1d>& layer2_gradient_y,
                  const cv::Mat1d& alpha_image,
                  const cv::Mat1d& alpha_gradient_x,
                  const cv::Mat1d& alpha_gradient_y,
                  const cv::Vec3d observed_color, const cv::Vec2i center_pixel,
                  const cv::Vec2i image_size)
      : layer1_image_(layer1_image),
        layer2_image_(layer2_image),
        alpha_image_(alpha_image),
        image_size_(image_size) {}

  bool operator()(const double* const layer1_flow,
                  const double* const layer2_flow, double* e) const {
    double layer1[3];
    double layer2[3];
    double alpha;
    double layer1_coord[2];
    double layer2_coord[2];
    layer1_coord[0] = center_pixel_[0] + layer1_flow[0];
    layer1_coord[1] = center_pixel_[1] + layer1_flow[1];
    layer2_coord[0] = center_pixel_[0] + layer2_flow[0];
    layer2_coord[1] = center_pixel_[1] + layer2_flow[1];
    if (layer1_coord[0] < 0 || layer1_coord[0] >= layer1_image_[0].cols ||
        layer1_coord[1] < 0 || layer1_coord[1] >= layer1_image_[0].rows ||
        layer2_coord[0] < 0 || layer2_coord[0] >= layer2_image_[0].cols ||
        layer2_coord[1] < 0 || layer2_coord[1] >= layer2_image_[0].rows) {
      const double penalty =
          1.0 + std::fabs(std::fmin(0, layer1_coord[0])) +
          std::fabs(std::fmin(0, layer1_coord[1])) +
          std::fabs(std::fmin(0, layer2_coord[0])) +
          std::fabs(std::fmin(0, layer2_coord[1])) +
          std::fabs(
              std::fmin(0, layer1_image_[0].cols - (1 + layer1_coord[0]))) +
          std::fabs(
              std::fmin(0, layer1_image_[0].rows - (1 + layer1_coord[1]))) +
          std::fabs(
              std::fmin(0, layer2_image_[0].cols - (1 + layer2_coord[0]))) +
          std::fabs(
              std::fmin(0, layer2_image_[0].rows - (1 + layer2_coord[1])));
      for (int c = 0; c < 3; c++) {
        e[c] = penalty * 9000.0f;
      }
      return true;
    }
    for (int c = 0; c < 3; c++) {
      layer1[c] =
          BilinearFetch(layer1_image_[c], layer1_coord[0], layer1_coord[1]);
      layer2[c] =
          BilinearFetch(layer2_image_[c], layer2_coord[0], layer2_coord[1]);
    }
    alpha = BilinearFetch(alpha_image_, layer1_coord[0], layer1_coord[1]);

    for (int c = 0; c < 3; c++) {
      e[c] = observed_color_[c] - (layer1[c] + alpha * layer2[c]);
    }
    return true;
  }

 private:
  const std::vector<cv::Mat1d>& layer1_image_;
  const std::vector<cv::Mat1d>& layer2_image_;
  const cv::Mat1d& alpha_image_;
  const cv::Vec3d observed_color_;
  const cv::Vec2i center_pixel_;
  const cv::Vec2i image_size_;
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

bool CompositeMotionRefiner::Optimize(const cv::Mat3b& layer1_img,
                                      const cv::Mat3b& layer2_img,
                                      const cv::Mat1f& alpha_img,
                                      const cv::Mat3b& composite,
                                      cv::Mat2f& layer1, cv::Mat2f& layer2,
                                      const int num_iterations) {
  CHECK_EQ(composite.size(), layer1.size());
  CHECK_EQ(composite.size(), layer2.size());

  std::vector<Eigen::Vector2i> index_to_coord_;
  cv::Mat1i coord_to_index_(cv::Size(composite.size()), -1);
  std::vector<double> parameters(composite.cols * composite.rows * 4);

  for (int y = 0; y < composite.rows; ++y) {
    for (int x = 0; x < composite.cols; ++x) {
      const int var_id = index_to_coord_.size();
      index_to_coord_.emplace_back(x, y);
      coord_to_index_(y, x) = var_id;
    }
  }

  ceres::Problem problem;

  // const double flow_smoothness_lambda = 0.1;

  // For each pixel in layer 1 flow:
  //    Create one gradient cost in X direction (with SoftL1Loss)
  //    Create one gradient cost in Y direction (with SoftL1Loss)
  // For each pixel in layer 2 flow:
  //    Create one gradient cost in X direction (with SoftL1Loss)
  //    Create one gradient cost in Y direction (with SoftL1Loss)

  cv::Mat3d layer1_image_double;
  cv::Mat3d layer2_image_double;
  cv::Mat1d alpha_image_double;
  layer1_img.convertTo(layer1_image_double, CV_64FC3, 1.0 / 255.0);
  layer2_img.convertTo(layer2_image_double, CV_64FC3, 1.0 / 255.0);
  alpha_img.convertTo(alpha_image_double, CV_64FC1);

  cv::Mat3d layer2_gradient_x(layer2_img.size(), 0);
  cv::Mat3d layer2_gradient_y(layer2_img.size(), 0);
  cv::Mat3d layer1_gradient_x(layer1_img.size(), 0);
  cv::Mat3d layer1_gradient_y(layer1_img.size(), 0);
  cv::Mat1d alpha_gradient_x(alpha_img.size(), 0);
  cv::Mat1d alpha_gradient_y(alpha_img.size(), 0);

  // Construct the gradient images
  for (int row = 0; row < composite.rows; row++) {
    for (int col = 0; col < composite.cols; col++) {
      if (col < composite.cols - 1) {
        layer1_gradient_x(row, col) =
            layer1_image_double(row, col + 1) - layer1_image_double(row, col);
        layer2_gradient_x(row, col) =
            layer2_image_double(row, col + 1) - layer2_image_double(row, col);
        alpha_gradient_x(row, col) =
            alpha_image_double(row, col + 1) - alpha_image_double(row, col);
      }
      if (row < composite.rows - 1) {
        layer1_gradient_y(row, col) =
            layer1_image_double(row + 1, col) - layer1_image_double(row, col);
        layer2_gradient_y(row, col) =
            layer2_image_double(row + 1, col) - layer2_image_double(row, col);
        alpha_gradient_y(row, col) =
            alpha_image_double(row + 1, col) - alpha_image_double(row, col);
      }
    }
  }

  std::vector<cv::Mat1d> layer2_gradient_x_split;
  std::vector<cv::Mat1d> layer2_gradient_y_split;
  std::vector<cv::Mat1d> layer1_gradient_x_split;
  std::vector<cv::Mat1d> layer1_gradient_y_split;
  std::vector<cv::Mat1d> layer1_image_double_split;
  std::vector<cv::Mat1d> layer2_image_double_split;

  cv::split(layer2_gradient_x, layer2_gradient_x_split);
  cv::split(layer2_gradient_y, layer2_gradient_y_split);
  cv::split(layer1_gradient_x, layer1_gradient_x_split);
  cv::split(layer1_gradient_y, layer1_gradient_y_split);
  cv::split(layer1_image_double, layer1_image_double_split);
  cv::split(layer2_image_double, layer2_image_double_split);

  cv::Mat1b mask(cv::Size(composite.cols, composite.rows), 0);
  for (int row = 0; row < composite.rows; row++) {
    for (int col = 0; col < composite.cols; col++) {
      const int center_pixel = coord_to_index_(row, col) * 4;

      if (cv::norm(layer1(row, col)) < 1e5 &&
          cv::norm(layer2(row, col)) < 1e5) {
        const cv::Vec2i coordinate(col, row);
        const cv::Vec3b observed_color = composite(row, col);
        const cv::Vec3d observed_color_double(observed_color[0] / 255.0,
                                              observed_color[1] / 255.0,
                                              observed_color[2] / 255.0);
        ceres::NumericDiffCostFunction<DataCostFunctor,
                                       ceres::NumericDiffMethodType::CENTRAL, 3,
                                       2, 2>* data_cost =
            new ceres::NumericDiffCostFunction<
                DataCostFunctor, ceres::NumericDiffMethodType::CENTRAL, 3, 2,
                2>(new DataCostFunctor(
                layer1_image_double_split, layer1_gradient_x_split,
                layer1_gradient_y_split, layer2_image_double_split,
                layer2_gradient_x_split, layer2_gradient_y_split,
                alpha_image_double, alpha_gradient_x, alpha_gradient_y,
                observed_color_double, coordinate,
                cv::Vec2i(composite.cols, composite.rows)));
        problem.AddResidualBlock(data_cost,
                                 new ceres::SoftLOneLoss(1.0 / 255.0),
                                 parameters.data() + center_pixel,
                                 parameters.data() + center_pixel + 2);
        mask(row, col) = 255;
      }

      // if (col < composite.cols - 1) {
      // const int right_pixel = coord_to_index_(row, col + 1) * 4;

      // ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>* layer1_cost
      // = new ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>( new
      // GradientCostFunctor(flow_smoothness_lambda, 2));
      // ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>* layer2_cost
      // = new ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>( new
      // GradientCostFunctor(flow_smoothness_lambda, 2));

      // problem.AddResidualBlock(
      // layer1_cost, new ceres::SoftLOneLoss(1.0 / 255.0),
      // parameters.data() + center_pixel, parameters.data() + right_pixel);
      // problem.AddResidualBlock(layer2_cost,
      // new ceres::SoftLOneLoss(1.0 / 255.0),
      // parameters.data() + center_pixel + 2,
      // parameters.data() + right_pixel + 2);
      //}

      // if (row < composite.rows - 1) {
      // const int bottom_pixel = coord_to_index_(row + 1, col) * 4;

      // ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>* layer1_cost
      // = new ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>( new
      // GradientCostFunctor(flow_smoothness_lambda, 2));
      // ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>* layer2_cost
      // = new ceres::AutoDiffCostFunction<GradientCostFunctor, 2, 2, 2>( new
      // GradientCostFunctor(flow_smoothness_lambda, 2));

      // problem.AddResidualBlock(
      // layer1_cost, new ceres::SoftLOneLoss(1.0 / 255.0),
      // parameters.data() + center_pixel, parameters.data() + bottom_pixel);
      // problem.AddResidualBlock(layer2_cost,
      // new ceres::SoftLOneLoss(1.0 / 255.0),
      // parameters.data() + center_pixel + 2,
      // parameters.data() + bottom_pixel + 2);
      //}
    }
  }

  cv::imshow("mask", mask);

  layer1.setTo(0, layer1 > 1e6);
  layer2.setTo(0, layer2 > 1e6);
  cv::Mat2f layer1_copy = layer1.clone();
  cv::Mat2f layer2_copy = layer2.clone();

  cv::Vec2i layer1_size(layer1_img.cols, layer1_img.rows);
  cv::Vec2i layer2_size(layer2_img.cols, layer2_img.rows);

  static const double kHalfWindowSize = 1.5;

  // Initialize the solution
  for (int row = 0; row < composite.rows; row++) {
    for (int col = 0; col < composite.cols; col++) {
      int pixel_index = coord_to_index_(row, col) * 4;
      cv::Vec2i xy(col, row);
      for (int c = 0; c < 2; c++) {
        parameters[pixel_index + c] = layer1(row, col)[c];
        const double lower_bound =
            std::fmin(std::fmax(-xy[c], layer1(row, col)[c] - kHalfWindowSize),
                      layer1_size[c] - (1 + xy[c]));
        const double upper_bound =
            std::fmax(std::fmin(layer1_size[c] - (1 + xy[c]),
                                layer1(row, col)[c] + kHalfWindowSize),
                      -xy[c]);

        problem.SetParameterLowerBound(parameters.data() + pixel_index, c,
                                       lower_bound);
        problem.SetParameterUpperBound(parameters.data() + pixel_index, c,
                                       upper_bound);
      }
      // problem.SetParameterUpperBound(parameters.data() + pixel_index, 0,
      // layer1_img.cols - (1 + col));
      // problem.SetParameterLowerBound(parameters.data() + pixel_index, 0,
      // -col); problem.SetParameterUpperBound(parameters.data() + pixel_index,
      // 1, layer1_img.rows - (1 + row));
      // problem.SetParameterLowerBound(parameters.data() + pixel_index, 1,
      // -row);
      for (int c = 0; c < 2; c++) {
        parameters[pixel_index + 2 + c] = layer2(row, col)[c];
        const double lower_bound =
            std::fmin(layer2_size[c] - (1 + xy[c]),
                      std::fmax(-xy[c], layer2(row, col)[c] - kHalfWindowSize));
        const double upper_bound =
            std::fmax(-xy[c], std::fmin(layer2_size[c] - (1 + xy[c]),
                                        layer2(row, col)[c] + kHalfWindowSize));
        problem.SetParameterLowerBound(parameters.data() + pixel_index, c,
                                       lower_bound);

        problem.SetParameterUpperBound(parameters.data() + pixel_index, c,
                                       upper_bound);
        CHECK_LE(lower_bound, upper_bound);
      }
      // problem.SetParameterUpperBound(parameters.data() + pixel_index + 2, 0,
      // layer1_img.cols - (1 + col));
      // problem.SetParameterLowerBound(parameters.data() + pixel_index + 2, 0,
      //-col);
      // problem.SetParameterUpperBound(parameters.data() + pixel_index + 2, 1,
      // layer1_img.rows - (1 + row));
      // problem.SetParameterLowerBound(parameters.data() + pixel_index + 2, 1,
      //-row);
    }
  }

  //    Optimize!
  LOG(INFO) << "Solving with Ceres...";
  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::CGNR;
  options.preconditioner_type = ceres::JACOBI;
  options.max_num_iterations = 500;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = 1e-6;
  options.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  // options.parameter_tolerance = 1e-20;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(ERROR) << summary.FullReport();

  LOG(INFO) << "Copying solution...";
  for (int row = 0; row < composite.rows; row++) {
    for (int col = 0; col < composite.cols; col++) {
      int pixel_index = coord_to_index_(row, col) * 4;
      for (int c = 0; c < 2; c++) {
        layer1(row, col)[c] = parameters[pixel_index + c];
      }
      for (int c = 0; c < 2; c++) {
        layer2(row, col)[c] = parameters[pixel_index + 2 + c];
      }
    }
  }

  // cv::imshow("original_l1", FlowToColor(layer1_copy));
  // cv::imshow("original_l2", FlowToColor(layer2_copy));
  // cv::imshow("new_l1", FlowToColor(layer1));
  // cv::imshow("new_l2", FlowToColor(layer2));
  // cv::imshow("delta_l1", FlowToColor(layer1 - layer1_copy));
  // cv::imshow("delta_l2", FlowToColor(layer2 - layer2_copy));
  // cv::waitKey();

  return true;
}  // namespace replay

}  // namespace replay
