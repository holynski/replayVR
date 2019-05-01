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
DEFINE_string(images_directory, "", "");
DEFINE_string(output_directory, "", "");
DEFINE_string(mesh, "", "");
DEFINE_string(window_mesh, "", "");
DEFINE_string(min_composite, "", "");
DEFINE_string(depth_cache, "", "");

struct ReconData {
  replay::Reconstruction* recon;
  replay::ImageCache* cache;
  std::string prefix;
  int smoothness_coefficient = 10;
};

int smoothStitchingFn(int pixel1, int pixel2, int label1, int label2,
                      void* extra_data) {
  ReconData* data = (ReconData*)extra_data;
  const cv::Mat3b& image1 =
      data->cache->Get(data->prefix + data->recon->GetCamera(label1).GetName());
  const cv::Mat3b& image2 =
      data->cache->Get(data->prefix + data->recon->GetCamera(label2).GetName());
  cv::Vec3b pix1_label1 = image1(pixel1);
  cv::Vec3b pix2_label1 = image1(pixel2);
  cv::Vec3b pix1_label2 = image2(pixel1);
  cv::Vec3b pix2_label2 = image2(pixel2);
  return (cv::norm(pix1_label1 - pix1_label2) +
          cv::norm(pix2_label1 - pix2_label2)) *
         data->smoothness_coefficient;
}

int smoothLabelFn(int pixel1, int pixel2, int label1, int label2,
                  void* extra_data) {
  ReconData* data = (ReconData*)extra_data;
  if (label1 == label2) {
    return 0;
  } else {
    return data->smoothness_coefficient;
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  replay::Reconstruction scene;
  CHECK(scene.Load(FLAGS_reconstruction));

  // Initialize caches. These are where the images are stored, so they don't
  // need to be recomputed each time.
  replay::ImageCache images(FLAGS_images_directory, 300);

  CHECK_GT(scene.NumCameras(), 0);

  LOG(INFO) << "Loading the mesh.";
  CHECK(replay::FileExists(FLAGS_mesh));
  replay::Mesh mesh;
  CHECK(mesh.Load(FLAGS_mesh));
  CHECK(mesh.NumTriangleFaces() > 0) << "Mesh is empty";

  LOG(ERROR) << "Initializing context";
  std::shared_ptr<replay::OpenGLContext> context =
      std::make_shared<replay::OpenGLContext>();
  CHECK(context->Initialize());

  // replay::ModelRenderer renderer(context);
  // CHECK(renderer.Initialize());
  LOG(ERROR) << "Uploading mesh";

  replay::ImageReprojector image_reprojector(context);

  replay::MinCompositorSequential min_compositor(context);
  LOG(ERROR) << "Uploading min composite images";
  CHECK(min_compositor.Initialize());
  int mesh_id = context->UploadMesh(mesh);
  context->BindMesh(mesh_id);

  replay::OpticalFlowAligner aligner(replay::OpticalFlowType::Greedy, context);
  replay::FuzzyMinDifference<cv::Vec3b> min_difference(context);

  replay::DepthMapRenderer depth_renderer(context);
  depth_renderer.Initialize();
  replay::Mesh window_mesh;
  int window_mesh_id = -1;
  if (replay::FileExists(FLAGS_window_mesh)) {
    CHECK(window_mesh.Load(FLAGS_window_mesh));
    window_mesh_id = context->UploadMesh(window_mesh);
  }

  /*
   * Establish the central viewpoint (average of all camera positions and
   * orientations), and set the FOV and image size to be relatively high
   */

  replay::Camera* central_view =
      scene.GetCamera(scene.NumCameras() / 2).Clone();

  LOG(INFO) << "Computing central viewpoint";
  // Compute the average position and orientation
  Eigen::Vector3d central_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d central_fwd = Eigen::Vector3d::Zero();
  Eigen::Vector3d central_up = Eigen::Vector3d::Zero();
  // central_fwd = mesh.GetMedianNormal().cast<double>();
  central_fwd = scene.GetCamera(scene.NumCameras() / 2).GetLookAt();
  if (central_fwd.dot(scene.GetCamera(scene.NumCameras() / 2).GetLookAt()) <
      0.f) {
    central_fwd *= -1;
  }

  // central_fwd = scene.GetCamera(0).GetLookAt() +
  // scene.GetCamera(scene.NumCameras() - 1).GetLookAt();
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

  // Set the central viewpoint position and orientation
  central_view->SetOrientationFromLookAtUpVector(central_fwd, central_up);
  central_view->SetPosition(central_position);

  // Double the horizontal FOV, since we're expecting the sequence to be a
  // horizontal sweep
  Eigen::Vector2d fov = central_view->GetFOV();
  central_view->SetFocalLengthFromFOV(Eigen::Vector2d(fov.x() * 2, fov.y()));
  // Double the image-width, so we don't lose angular resolution
  central_view->SetImageSize(Eigen::Vector2i(1920 * 2, 1080));

  // Save the frustum mesh for visualization
  replay::Mesh frustum_mesh = scene.CreateFrustumMesh();
  frustum_mesh.Append(mesh);
  frustum_mesh.Append(replay::Mesh::Frustum(*central_view));
  frustum_mesh.Save(replay::JoinPath(FLAGS_output_directory, "frustum.ply"));

  /*
   * Render the first layer images
   */

  context->BindMesh(mesh_id);

  replay::SimpleTimer timer;

  const Eigen::Vector2i image_size = central_view->GetImageSize();
  const int num_pixels = image_size.x() * image_size.y();
  const int num_labels = scene.NumCameras();
  GCoptimizationGridGraph* min_composite_gc =
      new GCoptimizationGridGraph(image_size.x(), image_size.y(), num_labels);
  cv::Mat3b min_composite_non_mrf(image_size.y(), image_size.x(),
                                  cv::Vec3b(255, 255, 255));
  replay::ImageStackAnalyzer::Options options;
  options.compute_max = false;
  options.compute_min = false;
  options.compute_median = false;
  replay::ImageStackAnalyzer first_layer_statistics(options);
  replay::FlowFromReprojection flow_reprojection(context);

  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    LOG(ERROR) << "Computing layer 1, frame (" << cam << "/"
               << scene.NumCameras() << "): " << timer.ElapsedTime() << "ms";
    const replay::Camera& camera = scene.GetCamera(cam);
    cv::Mat image = images.Get(camera.GetName()).clone();
    replay::ExposureAlignment::TransformImageExposure(
        image, camera.GetExposure(), Eigen::Vector3f(1, 1, 1), &image);

    // Reproject the image to the central viewpoint using the mesh
    cv::Mat3b reprojected;
    image_reprojector.SetImage(image);
    image_reprojector.SetSourceCamera(camera);
    image_reprojector.Reproject(*central_view, &reprojected);

    cv::Mat1b unknown_mask;
    cv::cvtColor(reprojected, unknown_mask, cv::COLOR_BGR2GRAY);
    unknown_mask = unknown_mask == 0;
    cv::dilate(unknown_mask, unknown_mask, cv::Mat());
    reprojected.setTo(cv::Vec3b(0, 0, 0), unknown_mask);
    first_layer_statistics.AddImage(reprojected, unknown_mask == 0);

    std::vector<GCoptimization::SparseDataCost> costs;
    for (int pixel = 0; pixel < reprojected.rows * reprojected.cols; pixel++) {
      if (unknown_mask(pixel) == 0) {
        GCoptimization::SparseDataCost cost;
        cost.site = pixel;
        cv::Vec3b color = reprojected(pixel);
        cost.cost = cv::norm(color);
        costs.emplace_back(cost);
      }
    }

    min_composite_gc->setDataCost(cam, costs.data(), costs.size());

    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "reprojected_" + camera.GetName()),
                reprojected);
    min_composite_non_mrf.copyTo(reprojected, unknown_mask);
    min_composite_non_mrf = cv::min(min_composite_non_mrf, reprojected);
  }

  cv::Mat3f variance_rgb = first_layer_statistics.GetVariance();
  cv::Mat1f variance;
  cv::cvtColor(variance_rgb, variance, cv::COLOR_BGR2GRAY);
  float variance_threshold = 20;
  cv::Mat3b mean = first_layer_statistics.GetMean();
  cv::Mat1b mean_gray;
  cv::cvtColor(mean, mean_gray, cv::COLOR_BGR2GRAY);
  cv::Mat1f mean_gradient;
  cv::Laplacian(mean_gray, mean_gradient, CV_32F, 15);
  cv::Mat1f scaled_variance;
  mean_gradient = cv::abs(mean_gradient);
  mean_gradient /= 10000000;
  // mean_gradient *= ;
  cv::divide(variance, mean_gradient + 1, scaled_variance);
  cv::Mat1b variance_mask = scaled_variance > variance_threshold;
  cv::erode(variance_mask, variance_mask, cv::Mat());
  cv::erode(variance_mask, variance_mask, cv::Mat());
  cv::erode(variance_mask, variance_mask, cv::Mat());
  cv::dilate(variance_mask, variance_mask, cv::Mat());
  cv::dilate(variance_mask, variance_mask, cv::Mat());
  cv::dilate(variance_mask, variance_mask, cv::Mat());

  cv::Mat3b min_composite = cv::imread(
      replay::JoinPath(FLAGS_images_directory, "../min_composite_mrf.png"));
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "first_layer_mask.png"),
              variance_mask);

  if (min_composite.empty()) {
    LOG(ERROR) << "Min composite empty...computing...";
    min_composite = cv::Mat3b(image_size.y(), image_size.x());
    cv::Mat3b labels(image_size.y(), image_size.x());
    // set up the needed data to pass to function for the data costs
    ReconData recon_data;
    recon_data.recon = &scene;
    replay::ImageCache reprojected_cache(FLAGS_output_directory, 300);
    recon_data.cache = &reprojected_cache;
    recon_data.prefix = "reprojected_";

    // smoothness comes from function pointer
    min_composite_gc->setSmoothCost(&smoothStitchingFn, &recon_data);

    std::unordered_map<int, cv::Vec3b> index_colors;
    for (int i = 0; i < scene.NumCameras(); i++) {
      index_colors[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
    }
    try {
      LOG(INFO) << "Before optimization energy is "
                << min_composite_gc->compute_energy();
      min_composite_gc->expansion(2);  // run expansion for 2 iterations. For
                                       // swap use gc->swap(num_iterations);
      LOG(INFO) << "After optimization energy is "
                << min_composite_gc->compute_energy();

      for (int pixel = 0; pixel < num_pixels; pixel++) {
        const int label = min_composite_gc->whatLabel(pixel);
        const cv::Mat3b& image = reprojected_cache.Get(
            recon_data.prefix + scene.GetCamera(label).GetName());
        min_composite(pixel) = image(pixel);
        labels(pixel) = index_colors[label];
      }

      delete min_composite_gc;
    } catch (GCException e) {
      e.Report();
    }
    cv::imwrite(replay::JoinPath(FLAGS_output_directory, "labels.png"), labels);
  }

  cv::imwrite(
      replay::JoinPath(FLAGS_output_directory, "min_composite_naive.png"),
      min_composite_non_mrf);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "min_composite_mrf.png"),
              min_composite);

  // We have the min-composite now, let's get the residuals

  const std::string sweep_cache_directory =
      replay::JoinPath(FLAGS_images_directory, "../sweep/");
  replay::PlaneSweep sweeper(context, sweep_cache_directory);
  replay::ImageReprojector image_reprojector2(context);
  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    LOG(ERROR) << "Computing layer 1 residual, frame (" << cam << "/"
               << scene.NumCameras() << "): " << timer.ElapsedTime() << "ms";
    const replay::Camera& camera = scene.GetCamera(cam);

    cv::Mat3b image = cv::imread(replay::JoinPath(
        FLAGS_output_directory, "reprojected_" + camera.GetName()));
    cv::Mat3b residual = min_difference.GetDifference(image, min_composite, 1);
    residual.setTo(cv::Vec3b(0, 0, 0), variance_mask == 0);
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "residual_" + camera.GetName()),
                residual);

    // Reproject the residual images back into the source viewpoints
    cv::Mat3b reprojected;
    image_reprojector2.SetImage(residual);
    image_reprojector2.SetSourceCamera(*central_view);
    image_reprojector2.Reproject(camera, &reprojected);
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "per_view_residual_" + camera.GetName()),
                reprojected);
    cv::Mat1b mask;
    cv::cvtColor(reprojected, mask, cv::COLOR_BGR2GRAY);
    mask = mask > 0;

    // Save them for later so we can estimate geometry
    sweeper.AddView(camera, reprojected, mask);
  }
  // Align the residual to find the dominant geometry
  LOG(ERROR) << "Computing depth for second layer.";
  replay::DepthMap base_depth;
  depth_renderer.GetDepthMap(*central_view, &base_depth);
  float first_layer_depth =
      base_depth.GetDepth(base_depth.Rows() / 2, base_depth.Cols() / 2);

  replay::PlaneSweepResult plane_sweep_result = sweeper.Sweep(
      *central_view, first_layer_depth * 2.0, first_layer_depth * 10.0, 100);
  float lowest_mean = FLT_MAX;
  float lowest_mean_index = -1;
  double min_cost = -1, max_cost = -1;
  for (const auto& cost_layer : plane_sweep_result.cost_volume) {
    if (min_cost <= 0) {
      cv::minMaxLoc(cost_layer.second, &min_cost, &max_cost);
    }
    cv::imwrite(
        replay::JoinPath(
            FLAGS_output_directory,
            "cost_" + std::to_string(cost_layer.first / first_layer_depth) +
                ".png"),
        cost_layer.second * 255.0 / max_cost);
    float mean = cv::mean(cost_layer.second, cost_layer.second > 0)[0];
    if (lowest_mean > mean) {
      lowest_mean = mean;
      lowest_mean_index = cost_layer.first;
    }
  }
  cv::imwrite(
      replay::JoinPath(FLAGS_output_directory, "second_layer_variance.png"),
      plane_sweep_result.cost_volume[lowest_mean_index] / 10);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "second_layer_mean.png"),
              plane_sweep_result.mean_images[lowest_mean_index]);

  GCoptimizationGridGraph* max_composite_gc =
      new GCoptimizationGridGraph(image_size.x(), image_size.y(), num_labels);
  cv::Mat3b max_composite_non_mrf(image_size.y(), image_size.x(),
                                  cv::Vec3b(0, 0, 0));

  plane_sweep_result.meshes[lowest_mean_index].Save(
      replay::JoinPath(FLAGS_output_directory, "second_layer_mesh.ply"));

  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    const replay::Camera& camera = scene.GetCamera(cam);
    cv::Mat3b residual_perview =
        cv::imread(replay::JoinPath(FLAGS_output_directory,

                                    "per_view_residual_" + camera.GetName()));
    context->BindMesh(plane_sweep_result.mesh_ids[lowest_mean_index]);
    cv::Mat3b reprojected;
    image_reprojector.SetImage(residual_perview);
    image_reprojector.SetSourceCamera(camera);
    image_reprojector.Reproject(*central_view, &reprojected);
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,

                                 "second_plane_" + camera.GetName()),
                reprojected);

    cv::Mat1b unknown_mask;
    cv::cvtColor(reprojected, unknown_mask, cv::COLOR_BGR2GRAY);
    unknown_mask = (unknown_mask == 0);
    cv::dilate(unknown_mask, unknown_mask, cv::Mat());
    cv::dilate(unknown_mask, unknown_mask, cv::Mat());
    reprojected.setTo(cv::Vec3b(0, 0, 0), unknown_mask);

    std::vector<GCoptimization::SparseDataCost> costs;
    for (int pixel = 0; pixel < reprojected.rows * reprojected.cols; pixel++) {
      if (unknown_mask(pixel) == 0) {
        GCoptimization::SparseDataCost cost;
        cost.site = pixel;
        cv::Vec3b color = reprojected(pixel);
        cost.cost = (cv::norm(cv::Vec3b(255, 255, 255)) - cv::norm(color));
        costs.emplace_back(cost);
      }
    }

    max_composite_gc->setDataCost(cam, costs.data(), costs.size());

    max_composite_non_mrf.copyTo(reprojected, unknown_mask);
    max_composite_non_mrf = cv::max(reprojected, max_composite_non_mrf);
  }

  cv::Mat3b max_composite = cv::imread(
      replay::JoinPath(FLAGS_images_directory, "../max_composite_mrf.png"));

  if (max_composite.empty()) {
    LOG(ERROR) << "Max composite empty...computing...";
    max_composite = cv::Mat3b(image_size.y(), image_size.x());
    cv::Mat3b labels(image_size.y(), image_size.x());
    // set up the needed data to pass to function for the data costs
    ReconData recon_data;
    recon_data.recon = &scene;
    replay::ImageCache reprojected_cache(FLAGS_output_directory, 300);
    recon_data.cache = &reprojected_cache;
    recon_data.smoothness_coefficient = 1000000;
    recon_data.prefix = "second_plane_";

    // smoothness comes from function pointer
    max_composite_gc->setSmoothCost(&smoothLabelFn, &recon_data);

    std::unordered_map<int, cv::Vec3b> index_colors;
    for (int i = 0; i < scene.NumCameras(); i++) {
      index_colors[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
    }
    try {
      LOG(INFO) << "Before optimization energy is "
                << max_composite_gc->compute_energy();
      max_composite_gc->expansion(2);  // run expansion for 2 iterations. For
                                       // swap use gc->swap(num_iterations);
      LOG(INFO) << "After optimization energy is "
                << max_composite_gc->compute_energy();

      for (int pixel = 0; pixel < num_pixels; pixel++) {
        const int label = max_composite_gc->whatLabel(pixel);
        const cv::Mat3b& image = reprojected_cache.Get(
            recon_data.prefix + scene.GetCamera(label).GetName());
        max_composite(pixel) = image(pixel);
        labels(pixel) = index_colors[label];
      }

      delete max_composite_gc;
    } catch (GCException e) {
      e.Report();
    }
    cv::imwrite(replay::JoinPath(FLAGS_output_directory, "labels_max.png"),
                labels);
  }

  cv::imwrite(
      replay::JoinPath(FLAGS_output_directory, "max_composite_naive.png"),
      max_composite_non_mrf);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "max_composite_mrf.png"),
              max_composite);

  cv::Mat1f alpha(min_composite.size(), 0.5);
  LOG(ERROR) << "Optimizing...";

  replay::ImageReprojector image_reprojector3(context);
  replay::ImageReprojector image_reprojector4(context);
  return 0;
  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    LOG(ERROR) << "Recomposing layer 1 + 2 (" << cam << "/"
               << scene.NumCameras() << "): " << timer.ElapsedTime() << "ms";
    const replay::Camera& camera = scene.GetCamera(cam);

    // Reproject the residual images back into the source viewpoints
    context->BindMesh(mesh_id);
    cv::Mat3b layer1_mean_reprojected;
    image_reprojector3.SetImage(min_composite);
    image_reprojector3.SetSourceCamera(*central_view);
    image_reprojector3.Reproject(camera, &layer1_mean_reprojected, 0.25);

    cv::Mat3b layer1_mask_reprojected;
    image_reprojector3.SetImage(variance_mask);
    image_reprojector3.SetSourceCamera(*central_view);
    image_reprojector3.Reproject(camera, &layer1_mask_reprojected);
    cv::Mat1b specular_mask;
    cv::cvtColor(layer1_mask_reprojected, specular_mask, cv::COLOR_BGR2GRAY);

    // image_reprojector4.Reproject(*central_view, &residual_central);
    // cv::imwrite(replay::JoinPath(FLAGS_output_directory,
    //"aligned_residual_" + camera.GetName()),
    // residual_central);

    cv::Mat3b composed = layer1_mean_reprojected + layer2_reprojected;
    layer1_mean_reprojected.copyTo(composed, specular_mask == 0);
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "composed_" + camera.GetName()),
                composed);
  }
}
