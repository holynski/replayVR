#pragma once

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <vector>

#include <replay/mesh/mesh.h>
#include <replay/rendering/depth_map_renderer.h>
#include <replay/rendering/image_reprojector.h>
#include <replay/sfm/reconstruction.h>
#include <replay/util/image_cache.h>

namespace replay {

class ExposureAlignment {
 public:
  enum LossFn {
    LOSS_L2,
    LOSS_CAUCHY,
    LOSS_HUBER,
  };

  struct Options {
    // Pixel values above which we consider to be overexposed, and treat
    // differently when optimizing.
    float overexposed_value = 0.9;

    // Scale of the optimization.
    float scale = 5.0;

    // Loss function used in optimization.
    LossFn lossfn = LossFn::LOSS_HUBER;

    // The minimum number of views that must see a particular point in order for
    // it to be considered a valid observation for optimization. Smaller values
    // will be faster, but may make the resulting estimate less consistent
    // across sequences of frames. Datasets with very sparse views may require
    // decreasing this value.
    int minimum_views_per_observation = 5;

    // The minimum number of observations to be used per frame. Smaller values
    // will be faster, but may make the resulting estimate more susceptible to
    // noisy observations in the form of incorrect geometry, camera sensor
    // noise, or motion blur. Datasets with very sparse meshes may require
    // decreasing this value.
    int minimum_observations_per_view = 400;

    // Solves for a single coefficient per image instead of three
    bool single_channel = false;
  };

  ExposureAlignment(std::shared_ptr<OpenGLContext> context,
                    const Options& options, const ImageCache& images,
                    Reconstruction* reconstruction);

  // Generates the exposure coefficients using the input mesh, reconstruction,
  // and image set. Runtime depends heavily on the options defined above. The
  // images are assumed to have sRGB gamma, and the returned exposure values are
  // transformed accordingly.
  std::vector<Eigen::Vector3f> GenerateExposureCoefficients();

  // Transforms an image (source) with initial exposure coefficient source_coeff
  // to match the exposure of target_coeff, and stores it in the output. The
  // output image may be the same as the input.
  static void TransformImageExposure(const cv::Mat& source,
                                     const Eigen::Vector3f& source_coeff,
                                     const Eigen::Vector3f& target_coeff,
                                     cv::Mat* output);

 private:
  Options options_;
  std::shared_ptr<OpenGLContext> context_;
  ImageReprojector reprojector_;
  DepthMapRenderer depth_renderer_;
  Reconstruction* reconstruction_;
  const ImageCache& images_;
};
}  // namespace replay

