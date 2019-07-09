#include <ceres/ceres.h>
#include <replay/flow/optical_flow_aligner.h>
#include <replay/flow/visualization.h>
#include <replay/multiview/layer_refiner.h>
#include <replay/rendering/opengl_context.h>
#include <Eigen/Sparse>

namespace replay {

namespace {}  // namespace

LayerRefiner::LayerRefiner(const int width, const int height)
    : width_(width),
      height_(height),
      current_row_(0),
      pixels_to_vars_(cv::Size(width_, height_), -1) {
  // Compute the sparse matrix using triplets.

  // Build index remaps
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const int var_id = vars_to_pixels_.size();
      vars_to_pixels_.emplace_back(x, y);
      pixels_to_vars_(y, x) = var_id;
    }
  }
}

// Do at multiple scales...
// how?

bool LayerRefiner::AddImage(const cv::Mat3b& image,
                            const cv::Mat2f& flow_to_layer1,
                            const cv::Mat2f& flow_to_layer2,
                            const cv::Mat1b& mask) {
  for (int y = 0; y < image.rows; ++y)
    for (int x = 0; x < image.cols; ++x) {
      if (flow_to_layer1(y, x)[0] == FLT_MAX ||
          flow_to_layer2(y, x)[0] == FLT_MAX || mask(y, x) == 0) {
        continue;
      }
      cv::Vec2f foreground_coordinates = cv::Vec2f(x, y) + flow_to_layer1(y, x);
      cv::Vec2f background_coordinates = cv::Vec2f(x, y) + flow_to_layer2(y, x);

      Eigen::Vector2i fg_floored(foreground_coordinates[0],
                                 foreground_coordinates[1]);
      Eigen::Vector2f fg_frac(foreground_coordinates[0] - fg_floored.x(),
                              foreground_coordinates[1] - fg_floored.y());

      Eigen::Vector2i bg_floored(background_coordinates[0],
                                 background_coordinates[1]);
      Eigen::Vector2f bg_frac(background_coordinates[0] - bg_floored.x(),
                              background_coordinates[1] - bg_floored.y());

      const int row_base = current_row_;
      current_row_ += 3;

      for (int c = 0; c < 3; ++c) {
        b_.push_back(image(y, x)[c]);
      }

      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          Eigen::Vector2f bgc =
              bg_floored.cast<float>() + Eigen::Vector2f(dx, dy);
          Eigen::Vector2f fgc =
              fg_floored.cast<float>() + Eigen::Vector2f(dx, dy);

          bgc.x() = std::min(bgc.x(), width_ - 1.0f);
          bgc.y() = std::min(bgc.y(), height_ - 1.0f);

          fgc.x() = std::min(fgc.x(), width_ - 1.0f);
          fgc.y() = std::min(fgc.y(), height_ - 1.0f);

          float bgw = (1.0f - std::abs(dy - bg_frac.y())) *
                      (1.0f - std::abs(dx - bg_frac.x()));
          float fgw = (1.0f - std::abs(dy - fg_frac.y())) *
                      (1.0f - std::abs(dx - fg_frac.x()));
          if (fgw <= 0.0f || bgw <= 0.0f) continue;

          int fgv = pixels_to_vars_(fgc.y(), fgc.x());
          int bgv = pixels_to_vars_(bgc.y(), bgc.x());

          for (int c = 0; c < 3; ++c) {
            const int row_id = row_base + c;
            const int fgv_c = fgv * 6 + c + 0;
            const int bgv_c = bgv * 6 + c + 3;
            triplets_.emplace_back(row_id, bgv_c, bgw);
            triplets_.emplace_back(row_id, fgv_c, fgw);
          }
        }
      }
    }
  return true;
}

bool LayerRefiner::Optimize(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img,
                            const int num_iterations) {
  for (int i = 0; i < num_iterations; ++i) {
    if (!GradientDescent(layer1_img, layer2_img)) {
      return false;
    }
    cv::imwrite("/Users/holynski/Downloads/debug/min_composite_optimized_" +
                    std::to_string(i) + ".png",
                layer1_img);
    cv::imwrite("/Users/holynski/Downloads/debug/max_composite_optimized_" +
                    std::to_string(i) + ".png",
                layer2_img);
  }
  return true;
}

bool LayerRefiner::GradientDescent(cv::Mat3b& layer1_img,
                                   cv::Mat3b& layer2_img) {
  if (layer1_img.rows != layer2_img.rows ||
      layer1_img.cols != layer2_img.cols) {
    LOG(ERROR) << "Layer sizes must be identical";
    return false;
  }

  // Take the current state of the linear system, and clone it.
  // The following constraints are goign to change at each L1 iteration, because
  // they depend on the initialization.
  std::vector<Eigen::Triplet<double>> triplets = triplets_;
  int current_row = current_row_;
  std::vector<double> bs = b_;

#if 1
  // Collect unary costs: Alleviate scale ambiguity with regularization ---
  // encourage solutions close to average grey.
  const float regularizer_lambda = 0.1f;
  for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i) {
    const int row_base = current_row;
    current_row += 6;

    for (int c = 0; c < 3; ++c) {
      const int fgr_c = row_base + c + 0;
      const int bgr_c = row_base + c + 3;
      const int fgv_c = i * 6 + c + 0;
      const int bgv_c = i * 6 + c + 3;
      triplets.emplace_back(fgr_c, fgv_c, regularizer_lambda);
      triplets.emplace_back(bgr_c, bgv_c, regularizer_lambda);
      bs.push_back(regularizer_lambda * 128.0f);
      bs.push_back(regularizer_lambda * 128.0f);
    }
  }
#endif

#if 1
  // Collect pairwise costs: Encourage sparse (L1) gradients on both layers.
  const float pairwise_lambda_fg = 10.0f;
  const float pairwise_lambda_bg = 10.0f;

  for (int y = 0; y < height_; ++y)
    for (int x = 0; x < width_; ++x) {
      const int this_id =
          pixels_to_vars_(std::min(height_ - 1, std::max(0, y + 0)),
                          std::min(width_ - 1, std::max(0, x + 0)));
      const int right_id =
          pixels_to_vars_(std::min(height_ - 1, std::max(0, y + 0)),
                          std::min(width_ - 1, std::max(0, x + 1)));
      const int bottom_id =
          pixels_to_vars_(std::min(height_ - 1, std::max(0, y + 1)),
                          std::min(width_ - 1, std::max(0, x + 0)));

      if (x < width_ - 1 && right_id >= 0) {
        const int row_base = current_row;
        current_row += 6;
        for (int c = 0; c < 3; ++c) {
          int fgr_t = this_id * 6 + 0 + c;
          int bgr_t = this_id * 6 + 3 + c;

          int fgr_r = right_id * 6 + 0 + c;
          int bgr_r = right_id * 6 + 3 + c;

          int fgr_c = row_base + 0 + c;
          int bgr_c = row_base + 3 + c;

          float prev_delta = 1.0f + std::abs(1.0f * layer1_img(y, x)[c] -
                                             1.0f * layer1_img(y, x + 1)[c]);
          float w = pairwise_lambda_fg / prev_delta;
          triplets.emplace_back(fgr_c, fgr_t, -w);
          triplets.emplace_back(fgr_c, fgr_r, w);
          bs.push_back(0.0);

          prev_delta = 1.0f + std::abs(1.0f * layer2_img(y, x)[c] -
                                       1.0f * layer2_img(y, x + 1)[c]);
          w = pairwise_lambda_bg / prev_delta;
          triplets.emplace_back(bgr_c, bgr_t, -w);
          triplets.emplace_back(bgr_c, bgr_r, w);
          bs.push_back(0.0);
        }
      }
      if (y < height_ - 1 & bottom_id >= 0) {
        const int row_base = current_row;
        current_row += 6;

        for (int c = 0; c < 3; ++c) {
          int fgr_t = this_id * 6 + 0 + c;
          int bgr_t = this_id * 6 + 3 + c;

          int fgr_b = bottom_id * 6 + 0 + c;
          int bgr_b = bottom_id * 6 + 3 + c;

          int fgr_c = row_base + 0 + c;
          int bgr_c = row_base + 3 + c;

          float prev_delta = 1.0f + std::abs(1.0f * layer1_img(y, x)[c] -
                                             1.0f * layer1_img(y + 1, x)[c]);
          float w = pairwise_lambda_fg / prev_delta;
          triplets.emplace_back(fgr_c, fgr_t, -w);
          triplets.emplace_back(fgr_c, fgr_b, w);
          bs.push_back(0.0);

          prev_delta = 1.0f + std::abs(1.0f * layer2_img(y, x)[c] -
                                       1.0f * layer2_img(y + 1, x)[c]);
          w = pairwise_lambda_bg / prev_delta;
          triplets.emplace_back(bgr_c, bgr_t, -w);
          triplets.emplace_back(bgr_c, bgr_b, w);
          bs.push_back(0.0);
        }
      }
    }
#endif
#if 1
  // Collect pairwise costs: Penalize correlated gradients between the FG and BG
  // layers.
  const float correlation_lambda = 10.0f;
  for (int y = 0; y < height_; ++y)
    for (int x = 0; x < width_; ++x) {
      const int this_id =
          pixels_to_vars_(std::min(height_ - 1, std::max(0, y + 0)),
                          std::min(width_ - 1, std::max(0, x + 0)));
      const int right_id =
          pixels_to_vars_(std::min(height_ - 1, std::max(0, y + 0)),
                          std::min(width_ - 1, std::max(0, x + 1)));
      const int bottom_id =
          pixels_to_vars_(std::min(height_ - 1, std::max(0, y + 1)),
                          std::min(width_ - 1, std::max(0, x + 0)));

      if (x < width_ - 1 && right_id >= 0) {
        const int row_base = current_row;
        current_row += 3;

        for (int c = 0; c < 3; ++c) {
          int fgr_t = this_id * 6 + 0 + c;
          int bgr_t = this_id * 6 + 3 + c;

          int fgr_r = right_id * 6 + 0 + c;
          int bgr_r = right_id * 6 + 3 + c;

          float prev_fgr_delta = layer1_img(y, x + 1)[c] - layer1_img(y, x)[c];
          float prev_bgr_delta = layer2_img(y, x + 1)[c] - layer2_img(y, x)[c];

          int row_id = row_base + c;
          float w = correlation_lambda;
          triplets.emplace_back(row_id, fgr_t, -w * prev_bgr_delta);
          triplets.emplace_back(row_id, fgr_r, w * prev_bgr_delta);
          triplets.emplace_back(row_id, bgr_t, -w * prev_fgr_delta);
          triplets.emplace_back(row_id, bgr_r, w * prev_fgr_delta);
          bs.push_back(w * prev_bgr_delta * prev_fgr_delta);
        }
      }

      if (y < height_ - 1 & bottom_id >= 0) {
        const int row_base = current_row;
        current_row += 3;

        for (int c = 0; c < 3; ++c) {
          int fgr_t = this_id * 6 + 0 + c;
          int bgr_t = this_id * 6 + 3 + c;

          int fgr_b = bottom_id * 6 + 0 + c;
          int bgr_b = bottom_id * 6 + 3 + c;

          float prev_fgr_delta = layer1_img(y + 1, x)[c] - layer1_img(y, x)[c];
          float prev_bgr_delta = layer2_img(y + 1, x)[c] - layer2_img(y, x)[c];

          int row_id = row_base + c;
          float w = correlation_lambda;
          triplets.emplace_back(row_id, fgr_t, -w * prev_bgr_delta);
          triplets.emplace_back(row_id, fgr_b, w * prev_bgr_delta);
          triplets.emplace_back(row_id, bgr_t, -w * prev_fgr_delta);
          triplets.emplace_back(row_id, bgr_b, w * prev_fgr_delta);
          bs.push_back(w * prev_bgr_delta * prev_fgr_delta);
        }
      }
    }
#endif

  LOG(INFO) << "Sparse matrix size: " << triplets.size() << std::endl;
  Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> A(
      current_row, vars_to_pixels_.size() * 6);
  A.setFromTriplets(triplets.begin(), triplets.end());
  Eigen::MatrixXd b(current_row, 1);
  for (int i = 0; i < current_row; ++i) {
    b(i, 0) = bs[i];
  }

  const Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> At =
      A.transpose();
  const Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> AtA =
      At * A;
  const Eigen::MatrixXd Atb = At * b;
  Eigen::MatrixXd solution(vars_to_pixels_.size() * 6, 1);
  Eigen::MatrixXd guess(vars_to_pixels_.size() * 6, 1);
  for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i) {
    for (int c = 0; c < 3; ++c) {
      Eigen::Vector2i coords = vars_to_pixels_[i];
      guess(i * 6 + c + 0) = layer1_img(coords.y(), coords.x())[c];
      guess(i * 6 + c + 3) = layer2_img(coords.y(), coords.x())[c];
    }
  }

  Eigen::ConjugateGradient<Eigen::SparseMatrix<double, 0, std::ptrdiff_t>,
                           Eigen::Lower | Eigen::Upper>
      solver;
  solver.setMaxIterations(250);
  solver.setTolerance(1e-6);
  solver.compute(AtA);
  solution = solver.solveWithGuess(Atb, guess);
  std::cout << "SOLVER ERROR: " << solver.error()
            << " ITERATIONS: " << solver.iterations() << std::endl;

  // layer1_img = cv::Mat3f::zeros(transparency_mask_.size());
  // layer2_img = cv::Mat3f::zeros(transparency_mask_.size());

  for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i)
    for (int c = 0; c < 3; ++c) {
      Eigen::Vector2i coords = vars_to_pixels_[i];
      layer1_img(coords.y(), coords.x())[c] =
          std::min(255.0, std::max(0.0, solution(i * 6 + c + 0)));
      layer2_img(coords.y(), coords.x())[c] =
          std::min(255.0, std::max(0.0, solution(i * 6 + c + 3)));
    }

  return true;
}

}  // namespace replay
