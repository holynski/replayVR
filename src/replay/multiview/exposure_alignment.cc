#include "replay/multiview/exposure_alignment.h"

#include <ceres/ceres.h>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "replay/depth_map/depth_map.h"

namespace replay {

namespace {

std::mutex thread_lock;

using namespace ceres;

size_t EigenVectorHash(const Eigen::Vector2i& vec) {
  return std::hash<int>()(vec[0] << 11 | vec[1]);
}

struct OverexposedFunctor {
  OverexposedFunctor(float lambda) : l(lambda) {}
  template <typename T>
  bool operator()(const T* const camera, const T* const point,
                  T* residual) const {
    residual[0] = camera[0] * point[0] - T(1.f);
    if (camera[0] * point[0] > T(1.f)) residual[0] *= T(l);
    return true;
  }

  float l;
};

struct CostFunctor {
  CostFunctor(float pixelvalue) : v(pixelvalue) {}
  template <typename T>
  bool operator()(const T* const camera, const T* const point,
                  T* residual) const {
    residual[0] = camera[0] * point[0] - T(v);
    return true;
  }
  float v;
};

const float lambda = 0.f;

CostFunction* Create(const float& pixelvalue, const float& overexposed_value) {
  if (pixelvalue >= overexposed_value) {
    return (new ceres::AutoDiffCostFunction<OverexposedFunctor, 1, 1, 1>(
        new OverexposedFunctor(lambda)));
  } else {
    return (new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1>(
        new CostFunctor(pixelvalue)));
  }
}

typedef std::unordered_map<int, Eigen::Vector3f> Correspondence;

std::vector<Eigen::Vector3f> Optimize(
    const std::vector<Correspondence>& observations, const int& num_images,
    const float& overexposed_value, const ExposureAlignment::LossFn& lossfn,
    const bool single_channel) {
  double scale = 5;
  std::vector<Eigen::Vector3f> coefficients(num_images);

  if (single_channel) {
    std::vector<double> channel_exposures(num_images, 1);
    std::vector<double> radiances(observations.size() * 3, 1);
    double cost = 0;
    ceres::Problem problem;
    for (int channel = 0; channel < 3; channel++) {
      for (int point = 0; point < observations.size(); point++) {
        for (auto iter = observations[point].begin();
             iter != observations[point].end(); iter++) {
          ceres::CostFunction* costfn;
          costfn = Create(iter->second[channel], overexposed_value);
          if (lossfn == ExposureAlignment::LossFn::LOSS_CAUCHY) {
            problem.AddResidualBlock(costfn, new ceres::CauchyLoss(scale),
                                     &channel_exposures[iter->first],
                                     &radiances[point * 3 + channel]);
          } else if (lossfn == ExposureAlignment::LossFn::LOSS_HUBER) {
            problem.AddResidualBlock(costfn, new ceres::HuberLoss(scale),
                                     &channel_exposures[iter->first],
                                     &radiances[point * 3 + channel]);
          } else {
            problem.AddResidualBlock(costfn, NULL,
                                     &channel_exposures[iter->first],
                                     &radiances[point * 3 + channel]);
          }
          problem.SetParameterLowerBound(&channel_exposures[iter->first], 0, 0);
        }
        problem.SetParameterLowerBound(&radiances[point * 3 + channel], 0, 0);
      }
    }
    problem.SetParameterBlockConstant(&(channel_exposures[0]));
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.use_inner_iterations = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    cost = std::max(cost, summary.final_cost);
    for (int i = 0; i < num_images; i++) {
      for (int channel = 0; channel < 3; channel++) {
        coefficients[i][channel] = static_cast<float>(channel_exposures[i]);
      }
    }
  } else {
    std::vector<double> channel_exposures(num_images, 1);
    std::vector<double> radiances(observations.size(), 1);
    double cost = 0;
    for (int channel = 0; channel < 3; channel++) {
      ceres::Problem problem;
      for (int point = 0; point < observations.size(); point++) {
        for (auto iter = observations[point].begin();
             iter != observations[point].end(); iter++) {
          ceres::CostFunction* costfn;
          costfn = Create(iter->second[channel], overexposed_value);
          if (lossfn == ExposureAlignment::LossFn::LOSS_CAUCHY) {
            problem.AddResidualBlock(costfn, new ceres::CauchyLoss(scale),
                                     &channel_exposures[iter->first],
                                     &radiances[point]);
          } else if (lossfn == ExposureAlignment::LossFn::LOSS_HUBER) {
            problem.AddResidualBlock(costfn, new ceres::HuberLoss(scale),
                                     &channel_exposures[iter->first],
                                     &radiances[point]);
          } else {
            problem.AddResidualBlock(costfn, NULL,
                                     &channel_exposures[iter->first],
                                     &radiances[point]);
          }
          problem.SetParameterLowerBound(&channel_exposures[iter->first], 0, 0);
        }
        problem.SetParameterLowerBound(&radiances[point], 0, 0);
      }

      problem.SetParameterBlockConstant(&(channel_exposures[0]));
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.minimizer_progress_to_stdout = true;
      options.use_inner_iterations = true;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.FullReport() << "\n";
      cost = std::max(cost, summary.final_cost);
      for (int i = 0; i < num_images; i++) {
        coefficients[i][channel] = static_cast<float>(channel_exposures[i]);
      }
    }
  }
  return coefficients;
}

Eigen::Vector2i RandomlySelectPixel(
    std::unordered_set<Eigen::Vector2i,
                       std::function<decltype(EigenVectorHash)>>*
        already_selected,
    int width, int height) {
  int random_index = static_cast<int>(std::rand() % (width * height));
  while (already_selected->count(
             Eigen::Vector2i(random_index % width, random_index / width)) > 0)
    random_index = (random_index + 1) % (width * height);
  Eigen::Vector2i retval(random_index % width, random_index / width);
  already_selected->insert(retval);
  return retval;
}

}  // namespace

ExposureAlignment::ExposureAlignment(std::shared_ptr<OpenGLContext> context,
                                     const ExposureAlignment::Options& options,
                                     const ImageCache& images,
                                     Reconstruction* reconstruction)
    : options_(options),
      context_(context),
      reprojector_(context_),
      depth_renderer_(context_),
      reconstruction_(reconstruction),
      images_(images) {
  CHECK(depth_renderer_.Initialize());
}

std::vector<Eigen::Vector3f> ExposureAlignment::GenerateExposureCoefficients() {
  std::vector<int> observations_per_view(reconstruction_->NumCameras(), 0);
  std::unordered_set<Eigen::Vector2i, std::function<decltype(EigenVectorHash)>>
      pixels(1920 * 1080, EigenVectorHash);
  std::vector<Correspondence> observations;
  while (true) {
    int first_camera_without_enough_observations = 0;
    while (observations_per_view[first_camera_without_enough_observations] >=
           options_.minimum_observations_per_view) {
      first_camera_without_enough_observations++;
    }
    LOG(INFO) << "Gathered observations for ("
              << first_camera_without_enough_observations << "/"
              << reconstruction_->NumCameras() << ") cameras.";
    if (first_camera_without_enough_observations >=
        reconstruction_->NumCameras()) {
      break;
    }

    const Camera& camera =
        reconstruction_->GetCamera(first_camera_without_enough_observations);
    const Eigen::Vector2i& image_size = camera.GetImageSize();

    std::vector<Eigen::Vector2i> random_pixels;
    random_pixels.clear();
    int start_index = observations.size();
    for (int i = 0; i < options_.minimum_observations_per_view * 8; i++) {
      random_pixels.push_back(
          RandomlySelectPixel(&pixels, image_size.x(), image_size.y()));
      observations.push_back(Correspondence());
    }

    // Eigen::Vector3d look_at1 = camera.GetLookAt();
    for (int i = 0; i < reconstruction_->NumCameras(); i++) {
      const Camera& camera_other = reconstruction_->GetCamera(i);
      reprojector_.SetSourceCamera(camera_other);
      // Eigen::Vector3d look_at2 = camera_other.GetLookAt();

      // TODO(holynski): Skip ahead if cameras don't overlap

      const std::string other_image_name = camera_other.GetName();
      const cv::Mat3b other_image = images_.Get(other_image_name);
      reprojector_.SetImage(other_image);
      cv::Mat3b output;
      CHECK(reprojector_.Reproject(camera, &output));
      cv::Mat1b gray;
      cv::cvtColor(output, gray, cv::COLOR_RGB2GRAY);
      cv::Mat1f gradient;
      cv::Laplacian(gray, gradient, CV_32F, 15);
      for (int p = 0; p < random_pixels.size(); p++) {
        if (gradient(random_pixels[p][1], random_pixels[p][0]) >
            options_.gradient_threshold) {
          continue;
        }
        cv::Vec3b color_b = output(random_pixels[p][1], random_pixels[p][0]);
        Eigen::Vector3f color(color_b[0] / 255.0, color_b[1] / 255.0,
                              color_b[2] / 255.0);
        if (color.norm() >= 0.05 && color.norm() <= 0.95) {
          color[0] = std::pow(color[0], 2.2);
          color[1] = std::pow(color[1], 2.2);
          color[2] = std::pow(color[2], 2.2);
          observations[p + start_index][i] = color;
        }
      }
    }
    int erased = 0;
    for (int p = observations.size() - 1; p >= start_index; p--) {
      if (observations[p].size() <
          std::min(options_.minimum_views_per_observation,
                   reconstruction_->NumCameras())) {
        observations.erase(observations.begin() + p);
        erased++;
        continue;
      }
      for (auto iter = observations[p].begin(); iter != observations[p].end();
           iter++) {
        observations_per_view[iter->first]++;
      }
    }
  }

  const std::vector<Eigen::Vector3f> coeffs = Optimize(
      observations, reconstruction_->NumCameras(), options_.overexposed_value,
      options_.lossfn, options_.single_channel);
  for (int i = 0; i < reconstruction_->NumCameras(); i++) {
    reconstruction_->GetCameraMutable(i)->SetExposure(coeffs[i]);
  }
  return coeffs;
}

void ExposureAlignment::TransformImageExposure(
    const cv::Mat& source, const Eigen::Vector3f& source_coeff,
    const Eigen::Vector3f& target_coeff, cv::Mat* output) {
  CHECK_NOTNULL(output);
  CHECK_EQ(output->cols, source.cols);
  CHECK_EQ(output->rows, source.rows);

  const int depth = source.depth();
  cv::Mat float_source = source.clone();
  if (depth != CV_32F && depth != CV_64F) {
    float_source.convertTo(float_source, CV_32F);
  }
  cv::pow(float_source, 2.2, float_source);

  cv::Scalar transform(target_coeff[0] / source_coeff[0],
                       target_coeff[1] / source_coeff[1],
                       target_coeff[2] / source_coeff[2]);
  cv::multiply(float_source, transform, *output);
  cv::pow(*output, 1.0 / 2.2, *output);
  if (depth != CV_32F && depth != CV_64F) {
    output->convertTo(*output, depth);
  }
}
}  // namespace replay
