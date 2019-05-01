#include <gflags/gflags.h>
#include <glog/logging.h>
#include <replay/camera/camera.h>
#include <replay/camera/pinhole_camera.h>
#include <replay/flow/flow_from_reprojection.h>
#include <replay/flow/optical_flow_aligner.h>
#include <replay/flow/visualization.h>
#include <replay/image/fuzzy_difference.h>
#include <replay/image/image_stack_analyzer.h>
#include <replay/image/sum_absolute_difference.h>
#include <replay/io/read_bundler.h>
#include <replay/io/read_capreal.h>
#include <replay/io/write_float_image.h>
#include <replay/multiview/exposure_alignment.h>
#include <replay/multiview/layer_refiner.h>
#include <replay/multiview/plane_sweep.h>
#include <replay/rendering/image_based_renderer.h>
#include <replay/rendering/image_reprojector.h>
#include <replay/rendering/max_compositor_sequential.h>
#include <replay/rendering/min_compositor_sequential.h>
#include <replay/rendering/model_renderer.h>
#include <replay/rendering/opengl_context.h>
#include <replay/sfm/reconstruction.h>
#include <replay/util/depth_cache.h>
#include <replay/util/filesystem.h>
#include <replay/util/image_cache.h>
#include <replay/util/strings.h>
#include <replay/util/timer.h>

#include <GCoptimization.h>
#include <Eigen/Geometry>

DEFINE_string(reconstruction, "", "");
DEFINE_string(first_layer_mesh, "", "");
DEFINE_string(second_layer_mesh, "", "");
DEFINE_string(images_directory, "", "");
DEFINE_string(output_directory, "", "");
DEFINE_string(window_mesh, "", "");
DEFINE_string(first_layer_image, "", "");
DEFINE_string(second_layer_image, "", "");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  replay::Reconstruction scene;
  CHECK(scene.Load(FLAGS_reconstruction));
  replay::ImageCache images(FLAGS_images_directory, 300);

  cv::Mat3b first_layer_image = cv::imread(FLAGS_first_layer_image);
  CHECK(!first_layer_image.empty());

  cv::Mat3b second_layer_image = cv::imread(FLAGS_second_layer_image);
  CHECK(!second_layer_image.empty());

  /*
   * Establish the central viewpoint (average of all camera positions and
   * orientations), and set the FOV and image size to be relatively high
   */
  replay::Camera* central_view =
      scene.GetCamera(scene.NumCameras() / 2).Clone();
  LOG(INFO) << "Computing central viewpoint";
  Eigen::Vector3d central_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d central_fwd = Eigen::Vector3d::Zero();
  Eigen::Vector3d central_up = Eigen::Vector3d::Zero();
  central_fwd = scene.GetCamera(scene.NumCameras() / 2).GetLookAt();
  if (central_fwd.dot(scene.GetCamera(scene.NumCameras() / 2).GetLookAt()) <
      0.f) {
    central_fwd *= -1;
  }
  Eigen::Vector3d left_vector =
      (scene.GetCamera(scene.NumCameras() - 1).GetPosition() -
       scene.GetCamera(0).GetPosition())
          .normalized();
  central_up = central_fwd.cross(left_vector);
  if (central_up.dot(scene.GetCamera(0).GetUpVector()) < 0) {
    central_up *= -1;
  }
  Eigen::Vector3d central_right = central_up.cross(central_fwd);
  double furthest_away_z = FLT_MAX;
  std::vector<double> x_positions(scene.NumCameras());
  std::vector<double> y_positions(scene.NumCameras());
  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    const Eigen::Vector3d& pos = scene.GetCamera(cam).GetPosition();
    furthest_away_z = std::min(pos.dot(central_fwd), furthest_away_z);
    x_positions[cam] = pos.dot(central_right);
    y_positions[cam] = pos.dot(central_up);
  }
  std::sort(x_positions.begin(), x_positions.end());
  std::sort(y_positions.begin(), y_positions.end());
  central_position += central_fwd * furthest_away_z;
  central_position += central_up * y_positions[y_positions.size() / 2];
  central_position += central_right * x_positions[x_positions.size() / 2];
  central_view->SetOrientationFromLookAtUpVector(central_fwd, central_up);
  central_view->SetPosition(central_position);
  Eigen::Vector2d fov = central_view->GetFOV();
  central_view->SetFocalLengthFromFOV(Eigen::Vector2d(fov.x() * 2, fov.y()));

  central_view->SetImageSize(Eigen::Vector2i(1920 * 2, 1080));
  central_view->SetImageSize(Eigen::Vector2i(1920 / 2, 1080 / 4));

  cv::resize(first_layer_image, first_layer_image,
             cv::Size(1920 / 2, 1080 / 4));
  cv::resize(second_layer_image, second_layer_image,
             cv::Size(1920 / 2, 1080 / 4));

  /*
   *
   *
   */

  replay::Mesh first_layer_mesh;
  CHECK(first_layer_mesh.Load(FLAGS_first_layer_mesh));
  CHECK(first_layer_mesh.NumTriangleFaces() > 0) << "Mesh is empty.";

  replay::Mesh second_layer_mesh;
  CHECK(second_layer_mesh.Load(FLAGS_second_layer_mesh));
  CHECK(second_layer_mesh.NumTriangleFaces() > 0) << "Mesh is empty.";

  LOG(ERROR) << "Initializing context";
  std::shared_ptr<replay::OpenGLContext> context =
      std::make_shared<replay::OpenGLContext>();
  CHECK(context->Initialize());

  replay::FlowFromReprojection flow_reprojection(context);
  int first_layer_id = context->UploadMesh(first_layer_mesh);
  int second_layer_id = context->UploadMesh(second_layer_mesh);

  replay::LayerRefiner refiner(first_layer_image.cols, first_layer_image.rows);
  cv::Mat1f alpha(first_layer_image.size(), -1.0);
  for (int cam = 0; cam < scene.NumCameras(); cam += 10) {
    LOG(ERROR) << "Adding image " << cam << "/" << scene.NumCameras();
    const replay::Camera& camera = scene.GetCamera(cam);

    context->BindMesh(first_layer_id);
    cv::Mat2f layer1_flow = flow_reprojection.Calculate(camera, *central_view);
    cv::Mat2f layer1_flow_reverse =
        flow_reprojection.Calculate(*central_view, camera);
    context->BindMesh(second_layer_id);
    cv::Mat2f layer2_flow = flow_reprojection.Calculate(camera, *central_view);
    double min, max;
    cv::minMaxLoc(layer2_flow, &min, &max);
    if (min == FLT_MAX) {
      LOG(ERROR) << "Skipping...flow all invalid";
      continue;
    }

    cv::Mat image = images.Get(camera.GetName()).clone();
    replay::ExposureAlignment::TransformImageExposure(
        image, camera.GetExposure(), Eigen::Vector3f(1, 1, 1), &image);

    // Warp both layers to this viewpoint so we can figure out an initial value
    // for alpha
    cv::Mat3b layer1_at_image =
        replay::OpticalFlowAligner::InverseWarp(first_layer_image, layer1_flow);

    cv::Mat3b layer2_at_image = replay::OpticalFlowAligner::InverseWarp(
        second_layer_image, layer2_flow);

    cv::Mat3f residual(image.size());
    cv::subtract(image, layer1_at_image, residual, cv::noArray(), CV_32F);
    residual.setTo(0, residual < 0);
    cv::Mat3f alpha_at_image_3_channel(image.size());
    cv::divide(residual, layer2_at_image, alpha_at_image_3_channel, 1, CV_32F);
    alpha_at_image_3_channel.setTo(0, layer2_at_image == 0);
    cv::Mat1f alpha_at_image;
    cv::cvtColor(alpha_at_image_3_channel, alpha_at_image, cv::COLOR_BGR2GRAY);

    cv::Mat1f alpha_guess = replay::OpticalFlowAligner::InverseWarp(
        alpha_at_image, layer1_flow_reverse);
    alpha_guess = 1 - alpha_guess;
    alpha_guess.copyTo(alpha, alpha <= 0.0);

    refiner.AddImage(image, layer1_flow, layer2_flow);

    layer1_flow.setTo(0, layer1_flow == FLT_MAX);
    layer2_flow.setTo(0, layer2_flow == FLT_MAX);

    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "layer1_flow_" + camera.GetName()),
                replay::FlowToColor(layer1_flow));
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "layer2_flow_" + camera.GetName()),
                replay::FlowToColor(layer2_flow));
  }

  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "initial_alpha.png"),
              alpha * 255);
  LOG(ERROR) << "Optimizing...";
  CHECK(refiner.Optimize(first_layer_image, second_layer_image, alpha));

  cv::Mat3f alpha_3c;
  cv::cvtColor(alpha, alpha_3c, cv::COLOR_GRAY2BGR);

  cv::Mat3f modulated(alpha.size());
  cv::multiply(second_layer_image, 1.0 - alpha_3c, modulated);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                               "max_composite_post_refine_modulated.png"),
              modulated);
  cv::multiply(first_layer_image, alpha_3c, modulated);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                               "min_composite_post_refine_modulated.png"),
              modulated);
  cv::imwrite(
      replay::JoinPath(FLAGS_output_directory, "min_composite_post_refine.png"),
      first_layer_image);
  cv::imwrite(
      replay::JoinPath(FLAGS_output_directory, "max_composite_post_refine.png"),
      second_layer_image);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "alpha.png"),
              alpha * 255);
  return 0;
}

