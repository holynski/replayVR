#include <ceres/ceres.h>
#include <replay/flow/optical_flow_aligner.h>
#include <replay/flow/visualization.h>
#include <replay/multiview/layer_refiner.h>
#include <replay/rendering/opengl_context.h>
#include <Eigen/Sparse>

namespace replay {

namespace {

static const float kEpsilon = 0.00001;

}  // namespace

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
                            const cv::Mat3b& layer1_img,
                            const cv::Mat3b& layer2_img,
                            const cv::Mat1f& alpha_img) {
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      if (flow_to_layer1(y, x)[0] == FLT_MAX ||
          flow_to_layer2(y, x)[0] == FLT_MAX) {
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
      if (fg_floored[0] < 0 || bg_floored[0] < 0 ||
          fg_floored[0] >= width_ - 1 || bg_floored[0] >= width_ - 1 ||
          fg_floored[1] < 0 || bg_floored[1] < 0 ||
          fg_floored[1] >= height_ - 1 || bg_floored[1] >= height_ - 1) {
        continue;
      }
      const int row_base = current_row_;
      current_row_ += 3;

      const cv::Vec3f observed =
          cv::Vec3f(image(y, x)[0] / 255.0, image(y, x)[1] / 255.0,
                    image(y, x)[2] / 255.0);
      ;
      const cv::Vec3f layer1 = cv::Vec3f(
          layer1_img(foreground_coordinates[1], foreground_coordinates[0])[0] /
              255.0,
          layer1_img(foreground_coordinates[1], foreground_coordinates[0])[1] /
              255.0,
          layer1_img(foreground_coordinates[1], foreground_coordinates[0])[2] /
              255.0);
      const cv::Vec3f layer2 = cv::Vec3f(
          layer2_img(background_coordinates[1], background_coordinates[0])[0] /
              255.0,
          layer2_img(background_coordinates[1], background_coordinates[0])[1] /
              255.0,
          layer2_img(background_coordinates[1], background_coordinates[0])[2] /
              255.0);
      const float alpha =
          alpha_img(foreground_coordinates[1], foreground_coordinates[0]);

      float w = 1.0f / std::sqrt(cv::norm(observed - layer1 - alpha * layer2,
                                          cv::NORM_L2SQR) +
                                 kEpsilon);

      for (int c = 0; c < 3; ++c) {
        b_.push_back(w * (observed[c] + alpha * layer2[c]));
      }
#if BILINEAR_TERMS
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

          const int av_c = fgv * 7 + 6;
          for (int c = 0; c < 3; ++c) {
            const int row_id = row_base + c;
            const int fgv_c = fgv * 7 + c + 0;
            const int bgv_c = bgv * 7 + c + 3;
            CHECK_LT(av_c, vars_to_pixels_.size() * 7);
            CHECK_LT(bgv_c, vars_to_pixels_.size() * 7);
            CHECK_LT(fgv_c, vars_to_pixels_.size() * 7);
            triplets_.emplace_back(row_id, av_c, bgw * w * layer2[c]);
            triplets_.emplace_back(row_id, bgv_c, bgw * w * alpha);
            triplets_.emplace_back(row_id, fgv_c, fgw * w);
          }
        }
      }
#else

      Eigen::Vector2f bgc(std::round(background_coordinates[0]),
                          std::round(background_coordinates[1]));
      Eigen::Vector2f fgc(std::round(foreground_coordinates[0]),
                          std::round(foreground_coordinates[1]));

      bgc.x() = std::max(0.0f, std::min(bgc.x(), width_ - 1.0f));
      bgc.y() = std::max(0.0f, std::min(bgc.y(), height_ - 1.0f));

      fgc.x() = std::max(0.0f, std::min(fgc.x(), width_ - 1.0f));
      fgc.y() = std::max(0.0f, std::min(fgc.y(), height_ - 1.0f));

      int fgv = pixels_to_vars_(fgc.y(), fgc.x());
      int bgv = pixels_to_vars_(bgc.y(), bgc.x());

      for (int c = 0; c < 3; ++c) {
        const int row_id = row_base + c;
        const int fgv_c = fgv * 7 + c + 0;
        const int bgv_c = bgv * 7 + c + 3;
        const int av_c = fgv * 7 + 6;
        CHECK_LT(av_c, vars_to_pixels_.size() * 7);
        CHECK_LT(bgv_c, vars_to_pixels_.size() * 7);
        CHECK_LT(fgv_c, vars_to_pixels_.size() * 7);
        triplets_.emplace_back(row_id, av_c, w * layer2[c]);
        triplets_.emplace_back(row_id, bgv_c, w * alpha);
        triplets_.emplace_back(row_id, fgv_c, w);
      }
#endif
    }
  }
  return true;
}

bool LayerRefiner::Optimize(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img,
                            cv::Mat1f& alpha, const int num_iterations) {
  for (int i = 0; i < num_iterations; ++i) {
    GradientDescent(layer1_img, layer2_img, alpha);
    cv::imwrite("/Users/holynski/Downloads/debug/min_composite_optimized_" +
                    std::to_string(i) + ".png",
                layer1_img);
    cv::imwrite("/Users/holynski/Downloads/debug/max_composite_optimized_" +
                    std::to_string(i) + ".png",
                layer2_img);
  }
  return true;
}

double LayerRefiner::GradientDescent(cv::Mat3b& layer1_img, cv::Mat3b& layer2_img,
                                   cv::Mat1f& alpha) {
  if (layer1_img.rows != layer2_img.rows ||
      layer1_img.cols != layer2_img.cols) {
    LOG(FATAL) << "Layer sizes must be identical";
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
  const float regularizer_lambda = 1.0f;
  for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i) {
    const int row_base = current_row;
    current_row += 7;

    const int ar_c = row_base + 6;
    const int av_c = i * 7 + 6;
    CHECK_LT(av_c, vars_to_pixels_.size() * 7);
    triplets.emplace_back(ar_c, av_c, regularizer_lambda);
    bs.push_back(regularizer_lambda * 0.5f);
    for (int c = 0; c < 3; ++c) {
      const int fgr_c = row_base + c + 0;
      const int bgr_c = row_base + c + 3;
      const int fgv_c = i * 7 + c + 0;
      const int bgv_c = i * 7 + c + 3;
      CHECK_LT(fgv_c, vars_to_pixels_.size() * 7);
      CHECK_LT(bgv_c, vars_to_pixels_.size() * 7);
      triplets.emplace_back(fgr_c, fgv_c, regularizer_lambda);
      triplets.emplace_back(bgr_c, bgv_c, regularizer_lambda);
      bs.push_back(regularizer_lambda * 0.5f);
      bs.push_back(regularizer_lambda * 0.5f);
    }
  }
#endif

#if 1
  // Collect pairwise costs: Encourage sparse (L1) gradients on both layers.
  const float pairwise_lambda_fg = 0.1f;
  const float pairwise_lambda_bg = 0.1f;
  const float pairwise_lambda_alpha = 1.0f;

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
        current_row += 7;

        int a_t = this_id * 7 + 6;
        int a_r = right_id * 7 + 6;
        int a_c = row_base + 7;
        CHECK_LT(a_t, vars_to_pixels_.size() * 7);
        CHECK_LT(a_r, vars_to_pixels_.size() * 7);
        triplets.emplace_back(a_c, a_t, -pairwise_lambda_alpha);
        triplets.emplace_back(a_c, a_r, pairwise_lambda_alpha);
        bs.push_back(0.0);

        for (int c = 0; c < 3; ++c) {
          int fgr_t = this_id * 7 + 0 + c;
          int bgr_t = this_id * 7 + 3 + c;

          int fgr_r = right_id * 7 + 0 + c;
          int bgr_r = right_id * 7 + 3 + c;

          int fgr_c = row_base + 0 + c;
          int bgr_c = row_base + 3 + c;

          float prev_delta =
              1.0f + std::abs(1.0f * layer1_img(y, x)[c] / 255.0 -
                              1.0f * layer1_img(y, x + 1)[c] / 255.0);
          float w = pairwise_lambda_fg / prev_delta;
          CHECK_LT(fgr_t, vars_to_pixels_.size() * 7);
          CHECK_LT(fgr_r, vars_to_pixels_.size() * 7);
          triplets.emplace_back(fgr_c, fgr_t, -w);
          triplets.emplace_back(fgr_c, fgr_r, w);
          bs.push_back(0.0);

          prev_delta = 1.0f + std::abs(1.0f * layer2_img(y, x)[c] / 255.0 -
                                       1.0f * layer2_img(y, x + 1)[c] / 255.0);
          w = pairwise_lambda_bg / prev_delta;
          CHECK_LT(bgr_t, vars_to_pixels_.size() * 7);
          CHECK_LT(bgr_r, vars_to_pixels_.size() * 7);
          triplets.emplace_back(bgr_c, bgr_t, -w);
          triplets.emplace_back(bgr_c, bgr_r, w);
          bs.push_back(0.0);
        }
      }
      if (y < height_ - 1 & bottom_id >= 0) {
        const int row_base = current_row;
        current_row += 7;

        int a_t = this_id * 7 + 6;
        int a_b = bottom_id * 7 + 6;
        int a_c = row_base + 7;
        CHECK_LT(a_t, vars_to_pixels_.size() * 7);
        CHECK_LT(a_b, vars_to_pixels_.size() * 7);
        triplets.emplace_back(a_c, a_t, -pairwise_lambda_alpha);
        triplets.emplace_back(a_c, a_b, pairwise_lambda_alpha);
        bs.push_back(0.0);

        for (int c = 0; c < 3; ++c) {
          int fgr_t = this_id * 7 + 0 + c;
          int bgr_t = this_id * 7 + 3 + c;

          int fgr_b = bottom_id * 7 + 0 + c;
          int bgr_b = bottom_id * 7 + 3 + c;

          int fgr_c = row_base + 0 + c;
          int bgr_c = row_base + 3 + c;

          float prev_delta =
              1.0f + std::abs(1.0f * layer1_img(y, x)[c] / 255.0 -
                              1.0f * layer1_img(y + 1, x)[c] / 255.0);
          float w = pairwise_lambda_fg / prev_delta;
          CHECK_LT(fgr_b, vars_to_pixels_.size() * 7);
          CHECK_LT(fgr_t, vars_to_pixels_.size() * 7);
          triplets.emplace_back(fgr_c, fgr_t, -w);
          triplets.emplace_back(fgr_c, fgr_b, w);
          bs.push_back(0.0);

          prev_delta = 1.0f + std::abs(1.0f * layer2_img(y, x)[c] / 255.0 -
                                       1.0f * layer2_img(y + 1, x)[c]) /
                                  255.0;
          w = pairwise_lambda_bg / prev_delta;
          CHECK_LT(bgr_b, vars_to_pixels_.size() * 7);
          CHECK_LT(bgr_t, vars_to_pixels_.size() * 7);
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
  const float correlation_lambda = 3000.0f;
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
          int fgr_t = this_id * 7 + 0 + c;
          int bgr_t = this_id * 7 + 3 + c;

          int fgr_r = right_id * 7 + 0 + c;
          int bgr_r = right_id * 7 + 3 + c;

          float prev_fgr_delta =
              layer1_img(y, x + 1)[c] / 255.0 - layer1_img(y, x)[c] / 255.0;
          float prev_bgr_delta =
              layer2_img(y, x + 1)[c] / 255.0 - layer2_img(y, x)[c] / 255.0;

          int row_id = row_base + c;
          float w = correlation_lambda;
          CHECK_LT(fgr_t, vars_to_pixels_.size() * 7);
          CHECK_LT(fgr_r, vars_to_pixels_.size() * 7);
          CHECK_LT(bgr_r, vars_to_pixels_.size() * 7);
          CHECK_LT(bgr_t, vars_to_pixels_.size() * 7);
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
          int fgr_t = this_id * 7 + 0 + c;
          int bgr_t = this_id * 7 + 3 + c;

          int fgr_b = bottom_id * 7 + 0 + c;
          int bgr_b = bottom_id * 7 + 3 + c;

          float prev_fgr_delta =
              layer1_img(y + 1, x)[c] / 255.0 - layer1_img(y, x)[c] / 255.0;
          float prev_bgr_delta =
              layer2_img(y + 1, x)[c] / 255.0 - layer2_img(y, x)[c] / 255.0;

          int row_id = row_base + c;
          float w = correlation_lambda;
          CHECK_LT(fgr_t, vars_to_pixels_.size() * 7);
          CHECK_LT(fgr_b, vars_to_pixels_.size() * 7);
          CHECK_LT(bgr_b, vars_to_pixels_.size() * 7);
          CHECK_LT(bgr_t, vars_to_pixels_.size() * 7);
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
      current_row, vars_to_pixels_.size() * 7);
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
  Eigen::MatrixXd solution(vars_to_pixels_.size() * 7, 1);
  Eigen::MatrixXd guess(vars_to_pixels_.size() * 7, 1);
  for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i) {
    Eigen::Vector2i coords = vars_to_pixels_[i];
    for (int c = 0; c < 3; ++c) {
      guess(i * 7 + c + 0) = layer1_img(coords.y(), coords.x())[c] / 255.0;
      guess(i * 7 + c + 3) = layer2_img(coords.y(), coords.x())[c] / 255.0;
    }
    guess(i * 7 + 6) = alpha(coords.y(), coords.x());
  }

  Eigen::ConjugateGradient<Eigen::SparseMatrix<double, 0, std::ptrdiff_t>,
                           Eigen::Lower | Eigen::Upper>
      solver;
  // solver.setMaxIterations(1);
  solver.setTolerance(1e-6);
  solver.compute(AtA);
#ifdef PROJECTED_GRAD
  double last_error = DBL_MAX;
  solver.setMaxIterations(1);
  while (last_error - solver.error() > 1e-6) {
    solution = solver.solveWithGuess(Atb, guess);
    for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i) {
      for (int c = 0; c < 3; ++c) {
        solution(i * 7 + c) =
            std::fmax(std::fmin(solution(i * 7 + c), 1.0), 0.0);
        solution(i * 7 + 3 + c) =
            std::fmax(std::fmin(solution(i * 7 + 3 + c), 1.0), 0.0);
      }
      solution(i * 7 + 6) = std::fmax(std::fmin(solution(i * 7 + 6), 1.0), 0.0);
    }
  }
#else
  solver.setMaxIterations(250);
  solution = solver.solveWithGuess(Atb, guess);
#endif
  std::cout << "SOLVER ERROR: " << solver.error()
            << " ITERATIONS: " << solver.iterations() << std::endl;

  // layer1_img = cv::Mat3f::zeros(transparency_mask_.size());
  // layer2_img = cv::Mat3f::zeros(transparency_mask_.size());

  for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i) {
    Eigen::Vector2i coords = vars_to_pixels_[i];
    for (int c = 0; c < 3; ++c) {
      layer1_img(coords.y(), coords.x())[c] =
          std::min(1.0, std::max(0.0, solution(i * 7 + c + 0))) * 255.0;
      layer2_img(coords.y(), coords.x())[c] =
          std::min(1.0, std::max(0.0, solution(i * 7 + c + 3))) * 255.0;
    }
    alpha(coords.y(), coords.x()) =
        std::min(1.0, std::max(0.0, solution(i * 7 + 6)));
  }

  return solver.error();
}

}  // namespace replay
