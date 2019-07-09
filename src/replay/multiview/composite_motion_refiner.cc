#include <ceres/ceres.h>
#include <replay/flow/visualization.h>
#include <replay/multiview/composite_motion_refiner.h>
#include <replay/rendering/opengl_context.h>
#include <replay/util/image.h>
#include <Eigen/Sparse>

namespace replay {

namespace {

static const float eps = 0.0001;
// static const int kNumElementsPerPixel = 4;
// static const int kFirstLayerFlowOffset = 0;
// static const int kSecondLayerFlowOffset = 2;

}  // namespace

CompositeMotionRefiner::CompositeMotionRefiner(const int width,
                                               const int height)
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

#if 0
{
  for (int y = 0; y < image.rows; ++y) {
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
  }
  return true;
}
#endif

bool CompositeMotionRefiner::Optimize(const cv::Mat3b& layer1_img,
                                      const cv::Mat3b& layer2_img,
                                      const cv::Mat1f& alpha_img,
                                      const cv::Mat3b& composite,
                                      cv::Mat2f& layer1, cv::Mat2f& layer2,
                                      const int num_iterations) {
  cv::imshow("layer1_before", replay::FlowToColor(layer1));
  cv::imshow("layer2_before", replay::FlowToColor(layer2));
  for (int i = 0; i < num_iterations; ++i) {
    if (!GradientDescent(layer1_img, layer2_img, alpha_img, composite, layer1,
                         layer2)) {
      return false;
    }
    cv::imshow("layer1", replay::FlowToColor(layer1));
    cv::imshow("layer2", replay::FlowToColor(layer2));
    cv::waitKey(1);
  }
  return true;
}

bool CompositeMotionRefiner::GradientDescent(const cv::Mat3b& layer1_img,
                                             const cv::Mat3b& layer2_img,
                                             const cv::Mat1f& alpha_img,
                                             const cv::Mat3b& composite,
                                             cv::Mat2f& layer1,
                                             cv::Mat2f& layer2) {
  if (layer1.rows != layer2.rows || layer1.cols != layer2.cols) {
    LOG(ERROR) << "Layer sizes must be identical";
    return false;
  }

  std::vector<Eigen::Triplet<double>> triplets = triplets_;
  int current_row = current_row_;
  std::vector<double> bs = b_;

  for (int y = 0; y < composite.rows; ++y) {
    for (int x = 0; x < composite.cols; ++x) {
      // I^t(x)
      const cv::Vec3b& composite_intensity = composite(y, x);

      // Vhat^t_O
      const cv::Vec2f& layer1_flow = layer1(y, x);

      // Vhat^t_B
      const cv::Vec2f& layer2_flow = layer2(y, x);

      // x
      const cv::Vec2f coord(x, y);

      const cv::Vec2f foreground_coordinates = coord + layer1_flow;
      const cv::Vec2f background_coordinates = coord + layer2_flow;

      if (foreground_coordinates[0] >= 0 &&
          foreground_coordinates[0] <= layer1_img.cols - 1 &&
          foreground_coordinates[1] >= 0 &&
          foreground_coordinates[1] <= layer1_img.rows - 1 &&
          background_coordinates[0] >= 0 &&
          background_coordinates[0] <= layer2_img.cols - 1 &&
          background_coordinates[1] >= 0 &&
          background_coordinates[1] <= layer2_img.rows - 1) {
        // Ibar_O(x)
        const cv::Vec3b layer1_intensity = BilinearFetch(
            layer1_img, foreground_coordinates[0], foreground_coordinates[1]);
        // Ibar_B(x)
        const cv::Vec3b layer2_intensity = BilinearFetch(
            layer2_img, background_coordinates[0], background_coordinates[1]);
        // Abar(x)
        const float alpha = BilinearFetch(alpha_img, foreground_coordinates[0],
                                          foreground_coordinates[1]);

        // \nabla Ibar_O(x)
        const cv::Vec3b layer1_gradient_x =
            BilinearGradient(layer1_img, foreground_coordinates[0],
                             foreground_coordinates[1], 0);
        const cv::Vec3b layer1_gradient_y =
            BilinearGradient(layer1_img, foreground_coordinates[0],
                             foreground_coordinates[1], 1);

        // \nabla Ibar_B(x)
        const cv::Vec3b layer2_gradient_x =
            BilinearGradient(layer2_img, background_coordinates[0],
                             background_coordinates[1], 0);
        const cv::Vec3b layer2_gradient_y =
            BilinearGradient(layer2_img, background_coordinates[0],
                             background_coordinates[1], 1);

        // \nabla Abar(x)
        const float alpha_gradient_x = BilinearGradient(
            alpha_img, foreground_coordinates[0], foreground_coordinates[1], 0);
        const float alpha_gradient_y = BilinearGradient(
            alpha_img, foreground_coordinates[0], foreground_coordinates[1], 1);

        // w_1(x)
        const float w_1 =
            1.0f / std::sqrt(cv::norm(composite_intensity - layer1_intensity -
                                          alpha * layer2_intensity,
                                      cv::NORM_L2SQR) +
                             eps);

        const int this_id = pixels_to_vars_(y, x) * 4;

        for (int c = 0; c < 3; c++) {
          // Index for foreground X flow
          const int v_o_x = this_id;
          // Index for foreground Y flow
          const int v_o_y = this_id + 1;
          // Index for background X flow
          const int v_b_x = this_id + 2;
          // Index for background Y flow
          const int v_b_y = this_id + 3;

          const int row = current_row + c;

          triplets.emplace_back(row, v_o_x,
                                w_1 * (layer2_intensity[c] * alpha_gradient_x +
                                       layer1_gradient_x[c]));
          triplets.emplace_back(row, v_o_y,
                                w_1 * (layer2_intensity[c] * alpha_gradient_y +
                                       layer1_gradient_y[c]));
          triplets.emplace_back(row, v_b_x,
                                w_1 * (alpha * layer2_gradient_x[c]));
          triplets.emplace_back(row, v_b_y,
                                w_1 * (alpha * layer2_gradient_y[c]));

          CHECK_EQ(w_1 * (layer2_intensity[c] * alpha_gradient_x +
                          layer1_gradient_x[c]),
                   w_1 * (layer2_intensity[c] * alpha_gradient_x +
                          layer1_gradient_x[c]));
          CHECK_EQ(w_1 * (layer2_intensity[c] * alpha_gradient_y +
                          layer1_gradient_y[c]),
                   w_1 * (layer2_intensity[c] * alpha_gradient_y +
                          layer1_gradient_y[c]));
          CHECK_EQ(w_1 * (alpha * layer2_gradient_x[c]),
                   w_1 * (alpha * layer2_gradient_x[c]));
          CHECK_EQ(w_1 * (alpha * layer2_gradient_y[c]),
                   w_1 * (alpha * layer2_gradient_y[c]));

          // Foreground intensity offset
          const cv::Vec2f vo_o =
              layer2_intensity[c] *
                  cv::Vec2f(alpha_gradient_x, alpha_gradient_y) +
              cv::Vec2f(layer1_gradient_x[c], layer1_gradient_y[c]);

          // Background intensity offset
          const cv::Vec2f vb_o =
              alpha * cv::Vec2f(layer2_gradient_x[c], layer2_gradient_y[c]);

          bs.emplace_back(w_1 *
                          (composite_intensity[c] - layer1_intensity[c] -
                           alpha * layer2_intensity[c] + vo_o.dot(layer1_flow) +
                           vb_o.dot(layer2_flow)));
        }
        current_row += 3;
      }

      const float smoothness_lambda = 0.5f;
      // w_2(x)
      const float w_2 = smoothness_lambda * 1.0 /
                        std::sqrt(cv::norm(layer1_flow, cv::NORM_L2SQR) + eps);

      // w_3(x)
      const float w_3 = smoothness_lambda * 1.0 /
                        std::sqrt(cv::norm(layer2_flow, cv::NORM_L2SQR) + eps);

      CHECK_EQ(w_2, w_2);
      CHECK_EQ(w_3, w_3);

      if (x + 1 < composite.cols) {
        const int this_id = pixels_to_vars_(y, x) * 4;
        const int right_id = pixels_to_vars_(y, x + 1) * 4;
        const int right_id_layer1_x = right_id;
        const int right_id_layer1_y = right_id + 1;
        const int right_id_layer2_x = right_id + 2;
        const int right_id_layer2_y = right_id + 3;

        const int this_id_layer1_x = this_id;
        // Index for foreground Y flow
        const int this_id_layer1_y = this_id + 1;
        // Index for background X flow
        const int this_id_layer2_x = this_id + 2;
        // Index for background Y flow
        const int this_id_layer2_y = this_id + 3;

        triplets.emplace_back(current_row, this_id_layer1_x, w_2);
        triplets.emplace_back(current_row, right_id_layer1_x, -w_2);
        bs.emplace_back(0);
        triplets.emplace_back(current_row + 1, this_id_layer1_y, w_2);
        triplets.emplace_back(current_row + 1, right_id_layer1_y, -w_2);
        bs.emplace_back(0);
        triplets.emplace_back(current_row + 2, this_id_layer2_x, w_3);
        triplets.emplace_back(current_row + 2, right_id_layer2_x, -w_3);
        bs.emplace_back(0);
        triplets.emplace_back(current_row + 3, this_id_layer2_y, w_3);
        triplets.emplace_back(current_row + 3, right_id_layer2_y, -w_3);
        bs.emplace_back(0);

        current_row += 4;
      }

      if (y + 1 < composite.rows) {
        const int this_id = pixels_to_vars_(y, x) * 4;
        const int bottom_id = pixels_to_vars_(y + 1, x) * 4;
        const int bottom_id_layer1_x = bottom_id;
        const int bottom_id_layer1_y = bottom_id + 1;
        const int bottom_id_layer2_x = bottom_id + 2;
        const int bottom_id_layer2_y = bottom_id + 3;

        const int this_id_layer1_x = this_id;
        // Index for foreground Y flow
        const int this_id_layer1_y = this_id + 1;
        // Index for background X flow
        const int this_id_layer2_x = this_id + 2;
        // Index for background Y flow
        const int this_id_layer2_y = this_id + 3;

        triplets.emplace_back(current_row, this_id_layer1_x, w_2);
        triplets.emplace_back(current_row, bottom_id_layer1_x, -w_2);
        bs.emplace_back(0);
        triplets.emplace_back(current_row + 1, this_id_layer1_y, w_2);
        triplets.emplace_back(current_row + 1, bottom_id_layer1_y, -w_2);
        bs.emplace_back(0);
        triplets.emplace_back(current_row + 2, this_id_layer2_x, w_3);
        triplets.emplace_back(current_row + 2, bottom_id_layer2_x, -w_3);
        bs.emplace_back(0);
        triplets.emplace_back(current_row + 3, this_id_layer2_y, w_3);
        triplets.emplace_back(current_row + 3, bottom_id_layer2_y, -w_3);
        bs.emplace_back(0);

        current_row += 4;
      }
    }
  }

  LOG(INFO) << "Sparse matrix size: " << triplets.size() << std::endl;
  Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> A(
      current_row, vars_to_pixels_.size() * 4);
  A.setFromTriplets(triplets.begin(), triplets.end());
  Eigen::MatrixXd b(current_row, 1);
  for (int i = 0; i < current_row; ++i) {
    CHECK_EQ(bs[i], bs[i]);
    b(i, 0) = bs[i];
  }

  const Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> At =
      A.transpose();
  const Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> AtA =
      At * A;
  const Eigen::MatrixXd Atb = At * b;
  Eigen::MatrixXd solution(vars_to_pixels_.size() * 4, 1);
  Eigen::MatrixXd guess(vars_to_pixels_.size() * 4, 1);
  for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i) {
    for (int c = 0; c < 2; ++c) {
      Eigen::Vector2i coords = vars_to_pixels_[i];
      guess(i * 4 + c + 0) = layer1(coords.y(), coords.x())[c];
      guess(i * 4 + c + 2) = layer2(coords.y(), coords.x())[c];
      if (guess(i * 4 + c + 0) > 5000) {
        guess(i * 4 + c + 0) = 0;
      }
      if (guess(i * 4 + c + 2) > 5000) {
        guess(i * 4 + c + 2) = 0;
      }
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

  CHECK_EQ(solver.error(), solver.error()) << "Solver error is NaN!";
  // layer1_img = cv::Mat3f::zeros(transparency_mask_.size());
  // layer2_img = cv::Mat3f::zeros(transparency_mask_.size());

  for (int i = 0; i < static_cast<int>(vars_to_pixels_.size()); ++i)
    for (int c = 0; c < 2; ++c) {
      Eigen::Vector2i coords = vars_to_pixels_[i];
      layer1(coords.y(), coords.x())[c] = solution(i * 4 + c + 0);
      layer2(coords.y(), coords.x())[c] = solution(i * 4 + c + 2);
    }

  return true;
}

}  // namespace replay
