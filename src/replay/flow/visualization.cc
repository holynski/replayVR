#include <glog/logging.h>
#include <math.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

namespace replay {
namespace {
int ncols = 0;
#define MAXCOLS 60
int colorwheel[MAXCOLS][3];

void setcols(int r, int g, int b, int k) {
  colorwheel[k][0] = r;
  colorwheel[k][1] = g;
  colorwheel[k][2] = b;
}

void makecolorwheel() {
  // relative lengths of color transitions:
  // these are chosen based on perceptual similarity
  // (e.g. one can distinguish more shades between red and yellow
  //  than between yellow and green)
  int RY = 15;
  int YG = 6;
  int GC = 4;
  int CB = 11;
  int BM = 13;
  int MR = 6;
  ncols = RY + YG + GC + CB + BM + MR;
  // printf("ncols = %d\n", ncols);
  if (ncols > MAXCOLS) exit(1);
  int i;
  int k = 0;
  for (i = 0; i < RY; i++) setcols(255, 255 * i / RY, 0, k++);
  for (i = 0; i < YG; i++) setcols(255 - 255 * i / YG, 255, 0, k++);
  for (i = 0; i < GC; i++) setcols(0, 255, 255 * i / GC, k++);
  for (i = 0; i < CB; i++) setcols(0, 255 - 255 * i / CB, 255, k++);
  for (i = 0; i < BM; i++) setcols(255 * i / BM, 0, 255, k++);
  for (i = 0; i < MR; i++) setcols(255, 0, 255 - 255 * i / MR, k++);
}

cv::Vec3b computeColor(float fx, float fy) {
  if (ncols == 0) makecolorwheel();

  cv::Vec3b retval;
  float rad = sqrt(fx * fx + fy * fy);
  float a = atan2(-fy, -fx) / M_PI;
  float fk = (a + 1.0) / 2.0 * (ncols - 1);
  int k0 = (int)fk;
  int k1 = (k0 + 1) % ncols;
  float f = fk - k0;
  // f = 0; // uncomment to see original color wheel
  for (int b = 0; b < 3; b++) {
    float col0 = colorwheel[k0][b] / 255.0;
    float col1 = colorwheel[k1][b] / 255.0;
    float col = (1 - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1 - rad * (1 - col);  // increase saturation with radius
    else
      col *= .75;  // out of range

    retval[2 - b] = (int)(255.0 * col);
  }
  return retval;
}
}  // namespace

cv::Mat3b FlowToColor(const cv::Mat2f& flow) {
  CHECK(!flow.empty());
  float max_radius = 0;
  for (int row = 0; row < flow.rows; row++) {
    for (int col = 0; col < flow.cols; col++) {
      if (flow(row, col)[0] == FLT_MAX || flow(row, col)[1] == FLT_MAX) {
        continue;
      }
      float radius = cv::norm(flow(row, col));
      max_radius = std::fmax(radius, max_radius);
    }
  }
  LOG(ERROR) << "Max radius: " << max_radius;
  cv::Mat3b color(flow.size());
  if (max_radius <= 0) {
    LOG(ERROR) << "Flow field is empty!";
    max_radius = 1;
  }
  for (int row = 0; row < flow.rows; row++) {
    for (int col = 0; col < flow.cols; col++) {
      const cv::Vec2f& flow_value = flow(row, col);
      if (flow_value != flow_value || flow_value[0] == FLT_MAX ||
          flow_value[1] == FLT_MAX) {
        color(row, col) = cv::Vec3b(0, 0, 0);
        continue;
      }
      color(row, col) =
          computeColor(flow_value[0] / max_radius, flow_value[1] / max_radius);
    }
  }
  return color;
}
}  // namespace replay
