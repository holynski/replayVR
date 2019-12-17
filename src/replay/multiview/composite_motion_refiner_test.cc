#include <gflags/gflags.h>
#include <stdio.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <string>

#include <gtest/gtest.h>
#include "replay/flow/optical_flow.h"
#include "replay/flow/visualization.h"
#include "replay/multiview/composite_motion_refiner.h"

DEFINE_string(test_img, "test1.png", "Name of test image file.");

namespace replay {
namespace {

std::string img_filename = REPLAY_DATA_DIR + std::string("/") + FLAGS_test_img;

#define ASSERT_IMG_EQ(img, img2)                        \
  ASSERT_EQ(img.cols, img2.cols);                       \
  ASSERT_EQ(img.rows, img2.rows);                       \
  ASSERT_EQ(img.channels(), img2.channels());           \
  for (int row = 0; row < img.rows; row++) {            \
    for (int col = 0; col < img.cols; col++) {          \
      for (int c = 0; c < img.channels(); c++) {        \
        ASSERT_EQ(img(row, col)[c], img2(row, col)[c]); \
      }                                                 \
    }                                                   \
  }

// Create a grid image
cv::Mat3b CreateGridImage(const int rows, const int cols, const int grid_size,
                          const cv::Vec3b& color1, const cv::Vec3b& color2) {
  cv::Mat3b image(rows, cols);
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      bool color_switch = ((row / grid_size) % 2) ^ ((col / grid_size) % 2);
      if (color_switch) {
        image(row, col) = color1;
      } else {
        image(row, col) = color2;
      }
    }
  }
  return image;
}

// A test that two composited images with an identity transformation converge to
// zero flow when the flow is initialized to zero.
TEST(CompositeMotionRefiner, IdentityTest) {
  cv::Mat3b layer1 = cv::imread(img_filename);
  ASSERT_FALSE(layer1.empty());
  layer1 = layer1 - 40;
  cv::imshow("layer1", layer1);
  const int rows = layer1.rows;
  const int cols = layer1.cols;
  cv::Mat3b layer2 = CreateGridImage(rows, cols, rows / 15,
                                     cv::Vec3b(40, 40, 40), cv::Vec3b(0, 0, 0));

  cv::Mat2f flow_layer1(rows, cols, cv::Vec2f(0, 0));
  cv::Mat2f flow_zero(rows, cols, cv::Vec2f(0, 0));
  cv::Mat2f flow_layer2(rows, cols, cv::Vec2f(0, 0));
  cv::Mat1f alpha(layer1.rows, layer1.cols, 1.0f);

  cv::Mat3b composite = layer2 + layer1;

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      cv::Point origin(col, row);
      cv::Vec2f layer1_flow = flow_layer1(row, col);
      cv::Vec2f layer2_flow = flow_layer2(row, col);
      for (int c = 0; c < 2; c++) {
        ASSERT_EQ(layer1_flow[c], 0.0);
        ASSERT_EQ(layer2_flow[c], 0.0);
      }
      cv::Point layer1_pixel =
          origin + cv::Point(layer1_flow[0], layer1_flow[1]);
      cv::Point layer2_pixel =
          origin + cv::Point(layer2_flow[0], layer2_flow[1]);
      ASSERT_EQ(layer1_pixel, origin);
      ASSERT_EQ(layer2_pixel, origin);
      cv::Vec3b observed_color = composite(row, col);
      cv::Vec3b layer1_color = layer1(layer1_pixel);
      cv::Vec3b layer2_color = layer2(layer2_pixel);
      float a = alpha(layer1_pixel);

      for (int c = 0; c < 3; c++) {
        ASSERT_EQ(layer1_color[c], layer1(row, col)[c]);
        ASSERT_EQ(layer2_color[c], layer2(row, col)[c]);
      }
      ASSERT_EQ(a, alpha(row, col));
      cv::Vec3b composite_pixel = layer1_color + a * layer2_color;
      for (int c = 0; c < 3; c++) {
        ASSERT_EQ(composite_pixel[c], observed_color[c]);
      }
    }
  }

  CompositeMotionRefiner::Optimize(layer1, layer2, alpha, composite,
                                   flow_layer1, flow_layer2, 5);

  ASSERT_IMG_EQ(flow_layer1, flow_zero);
  ASSERT_IMG_EQ(flow_layer2, flow_zero);
}

// A test that two composited images with an identity transformation converge to
// zero flow when the flow is initialized to something else.
TEST(CompositeMotionRefiner, IdentityBadInitialTest) {
  cv::Mat3b layer1 = cv::imread(img_filename);
  ASSERT_FALSE(layer1.empty());
  layer1 = layer1 - 40;
  const int rows = layer1.rows;
  const int cols = layer1.cols;
  cv::Mat3b layer2 = CreateGridImage(rows, cols, 8, cv::Vec3b(40, 40, 40),
                                     cv::Vec3b(0, 0, 0));
  cv::GaussianBlur(layer2, layer2, cv::Size(21, 21), 2, 2);

  cv::Mat2f flow_layer1(rows, cols, cv::Vec2f(0, 0));
  cv::Mat2f flow_zero(rows, cols, cv::Vec2f(0, 0));
  cv::Mat2f flow_layer2(rows, cols, cv::Vec2f(0, 1));
  cv::Mat1f alpha(layer1.rows, layer1.cols, 1.0f);

  cv::Mat3b composite = layer2 + layer1;
  cv::imshow("layer2", layer2);
  cv::imshow("layer1", layer1);
  cv::imshow("composite", composite);
  cv::waitKey();

  CompositeMotionRefiner::Optimize(layer1, layer2, alpha, composite,
                                   flow_layer1, flow_layer2, 5);

  cv::Mat1f residual(flow_layer1.size(), 0.0);
  cv::Mat3b residual_image(flow_layer1.size(), cv::Vec3b(0, 0, 0));
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      cv::Point origin(col, row);
      cv::Vec2f layer1_flow = flow_layer1(row, col);
      cv::Vec2f layer2_flow = flow_layer2(row, col);
      cv::Point layer1_pixel =
          origin + cv::Point(layer1_flow[0], layer1_flow[1]);
      cv::Point layer2_pixel =
          origin + cv::Point(layer2_flow[0], layer2_flow[1]);
      cv::Vec3b observed_color = composite(row, col);
      cv::Vec3b layer1_color = layer1(layer1_pixel);
      cv::Vec3b layer2_color = layer2(layer2_pixel);
      float a = alpha(layer1_pixel);
      cv::Vec3b composite_pixel = layer1_color + a * layer2_color;
      for (int c = 0; c < 3; c++) {
        residual_image(row, col)[c] =
            std::max(composite_pixel[c] - observed_color[c],
                     observed_color[c] - composite_pixel[c]);
      }

      residual(row, col) = std::fabs(flow_layer1(row, col)[1]);
      // EXPECT_EQ(flow_layer1(row,col), cv::Vec2f(0,0));
    }
  }
  cv::imshow("residual", residual);
  cv::imshow("residual_img", residual_image);
  cv::imshow("new_flow", FlowToColor(flow_layer2));
  cv::waitKey();

  double min_val = DBL_MAX;
  double max_val = -DBL_MAX;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      min_val = std::fmin(flow_layer2(row, col)[1], min_val);
      max_val = std::fmax(flow_layer2(row, col)[1], max_val);
    }
  }
  LOG(ERROR) << "Min/max: " << min_val << "/" << max_val;
  ASSERT_IMG_EQ(flow_layer2, flow_zero);
  ASSERT_IMG_EQ(flow_layer1, flow_zero);
  EXPECT_TRUE(true);
}

}  // namespace
}  // namespace replay
