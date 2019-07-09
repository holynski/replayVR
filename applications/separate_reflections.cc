#include <gflags/gflags.h>
#include <glog/logging.h>
#include <replay/camera/camera.h>
#include <replay/camera/pinhole_camera.h>
#include <replay/flow/flow_from_reprojection.h>
#include <replay/flow/greedy_flow.h>
#include <replay/flow/optical_flow_aligner.h>
#include <replay/flow/visualization.h>
#include <replay/geometry/plane.h>
#include <replay/image/colormap.h>
#include <replay/image/fuzzy_difference.h>
#include <replay/image/image_stack_analyzer.h>
#include <replay/image/sum_absolute_difference.h>
#include <replay/io/read_bundler.h>
#include <replay/io/read_capreal.h>
#include <replay/io/write_float_image.h>
#include <replay/multiview/composite_motion_refiner.h>
#include <replay/multiview/exposure_alignment.h>
#include <replay/multiview/layer_refiner.h>
#include <replay/multiview/plane_sweep.h>
#include <replay/multiview/reflection_segmenter.h>
#include <replay/rendering/image_based_renderer.h>
#include <replay/rendering/image_reprojector.h>
#include <replay/rendering/max_compositor_sequential.h>
#include <replay/rendering/min_compositor_sequential.h>
#include <replay/rendering/model_renderer.h>
#include <replay/rendering/opengl_context.h>
#include <replay/sfm/reconstruction.h>
#include <replay/sfm/video_tracker.h>
#include <replay/util/depth_cache.h>
#include <replay/util/filesystem.h>
#include <replay/util/image_cache.h>
#include <replay/util/progress_bar.h>
#include <replay/util/strings.h>
#include <replay/util/timer.h>

#include <Eigen/Geometry>

DEFINE_string(reconstruction, "", "");
DEFINE_string(partition_image, "", "");
DEFINE_string(images_directory, "", "");
DEFINE_string(output_directory, "", "");
DEFINE_string(mesh, "", "");
DEFINE_string(window_mesh, "", "");
DEFINE_string(min_composite, "", "");
DEFINE_string(depth_cache, "", "");

static const int kSkipFrames = 3;
static const int kMaxFrames = 30;

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

  replay::ImageReprojector image_reprojector(context);

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
  cv::Mat3b min_composite(image_size.y(), image_size.x(),
                          cv::Vec3b(255, 255, 255));
  replay::ImageStackAnalyzer::Options options;
  options.compute_max = false;
  options.compute_min = false;
  options.compute_median = false;
  replay::ImageStackAnalyzer first_layer_statistics(options);

  // Get initial min-composite

  for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
       cam += kSkipFrames) {
    replay::PrintProgress(
        cam + 1, std::min(scene.NumCameras(), kMaxFrames), "Computing layer 1",
        "- " + std::to_string(static_cast<int>(timer.ElapsedTime())) +
            " ms / frame");
    const replay::Camera& camera = scene.GetCamera(cam);
    cv::Mat image = images.Get(camera.GetName()).clone();
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "input_view_" + camera.GetName()),
                image);
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

    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "reprojected_" + camera.GetName()),
                reprojected);

    min_composite.copyTo(reprojected, unknown_mask);
    min_composite = cv::min(min_composite, reprojected);
  }

  // Get the flow resulting from the geometry
  std::unordered_map<int, cv::Mat2f> flows_to_layer1;
  std::unordered_map<int, cv::Mat2f> flows_from_layer1;
  std::unordered_map<int, cv::Mat2f> flows_to_layer2;
  std::unordered_map<int, cv::Mat2f> flows_from_layer2;

  replay::FlowFromReprojection flow_from_geometry(context);
  for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
       cam += kSkipFrames) {
    const replay::Camera& camera = scene.GetCamera(cam);
    context->BindMesh(mesh_id);
    cv::Mat2f flow_from_layer1 =
        flow_from_geometry.Calculate(*central_view, camera);
    flows_from_layer1[cam] = flow_from_layer1;
    cv::Mat2f flow_to_layer1 =
        flow_from_geometry.Calculate(camera, *central_view);
    flows_to_layer1[cam] = flow_to_layer1;
  }

  cv::Mat3b mean_composite = first_layer_statistics.GetMean();
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "min_composite_0.png"),
              min_composite);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "mean_composite_.png"),
              mean_composite);
  replay::GreedyFlow greedy_flow(context, 5);
  // static const int kNumLayer1MotionIterations = 1;
  // for (int it = 0; it < kNumLayer1MotionIterations; it++) {
  // LOG(ERROR) << "Refining motion, iteration " << it;
  //// Refine the first-layer refinement and min-composite
  // cv::Mat3b min_composite_refined(image_size.y(), image_size.x(),
  // cv::Vec3b(255, 255, 255));
  // cv::Mat3f variance_rgb = first_layer_statistics.GetVariance();
  // cv::Mat1f variance;
  // cv::cvtColor(variance_rgb, variance, cv::COLOR_BGR2GRAY);
  // replay::ImageStackAnalyzer first_layer_statistics_refined(options);

  // for (int cam = 0; cam < scene.NumCameras(); cam += kSkipFrames) {
  // const replay::Camera& camera = scene.GetCamera(cam);
  // const cv::Mat& image = images.Get(camera.GetName());
  // cv::Mat2f refined_flow = flows_from_layer1[cam].clone();
  // greedy_flow.calc(mean_composite, image, refined_flow);
  // cv::imshow("refined_backward",
  // replay::FlowToColor(refined_flow - flows_from_layer1[cam]));
  // flows_from_layer1[cam] = refined_flow;

  // cv::waitKey(1);
  // cv::Mat3b reprojected =
  // replay::OpticalFlowAligner::InverseWarp(image, refined_flow);
  // cv::imwrite(replay::JoinPath(FLAGS_output_directory,
  //"refined_reprojected_" + camera.GetName()),
  // reprojected);

  // cv::Mat1b unknown_mask;
  // cv::cvtColor(reprojected, unknown_mask, cv::COLOR_BGR2GRAY);
  // unknown_mask = unknown_mask == 0;
  // cv::dilate(unknown_mask, unknown_mask, cv::Mat());
  // reprojected.setTo(cv::Vec3b(0, 0, 0), unknown_mask);
  // first_layer_statistics_refined.AddImage(reprojected, unknown_mask == 0);
  // min_composite_refined.copyTo(reprojected, unknown_mask);
  // min_composite_refined = cv::min(min_composite_refined, reprojected);

  // refined_flow = flows_to_layer1[cam].clone();
  // greedy_flow.calc(image, min_composite_refined, refined_flow);
  // cv::imshow("refinement",
  // replay::FlowToColor(refined_flow - flows_to_layer1[cam]));
  // cv::imshow("refined_forward", replay::FlowToColor(refined_flow));
  // flows_to_layer1[cam] = refined_flow;
  //}

  // cv::Mat1b mean_gray;
  // cv::cvtColor(mean_composite, mean_gray, cv::COLOR_BGR2GRAY);
  // cv::Mat3b mean_composite_refined =
  // first_layer_statistics_refined.GetMean();

  // cv::imwrite(
  // replay::JoinPath(FLAGS_output_directory,
  //"min_composite_" + std::to_string(it + 1) + ".png"),
  // min_composite_refined);
  // cv::imwrite(
  // replay::JoinPath(FLAGS_output_directory,
  //"mean_composite_" + std::to_string(it + 1) + ".png"),
  // mean_composite_refined);
  // cv::imshow("previous", min_composite);
  // cv::imshow("current", min_composite_refined);
  // cv::waitKey(1);

  // min_composite = min_composite_refined;
  // mean_composite = first_layer_statistics_refined.GetMean();
  //}

  // We have the min-composite now, let's get the residuals

  const std::string sweep_cache_directory =
      replay::JoinPath(FLAGS_images_directory, "../sweep/");
  replay::PlaneSweep sweeper(context, sweep_cache_directory);
  replay::ImageReprojector image_reprojector2(context);

  replay::VideoTrackerOptions tracker_options;
  tracker_options.distance_between_keypoints = 12;
  tracker_options.max_points = 15000;
  tracker_options.min_points = 15000;
  replay::VideoTracker tracker(tracker_options);
  for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
       cam += kSkipFrames) {
    replay::PrintProgress(
        cam + 1, std::min(scene.NumCameras(), kMaxFrames), "Computing residual",
        "- " + std::to_string(static_cast<int>(timer.ElapsedTime())) +
            " ms / frame");
    const replay::Camera& camera = scene.GetCamera(cam);
    const cv::Mat& image = images.Get(camera.GetName());

    // Reproject the residual images back into the source viewpoints
    cv::Mat3b reprojected;
    image_reprojector2.SetImage(min_composite);
    image_reprojector2.SetSourceCamera(*central_view);
    context->SetClearColor(Eigen::Vector3f(1.0, 1.0, 1.0));
    image_reprojector2.Reproject(camera, &reprojected);
    context->SetClearColor(Eigen::Vector3f(0.0, 0.0, 0.0));
    cv::Mat3b residual = min_difference.GetDifference(image, reprojected, 1);
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "per_view_residual_0_" + camera.GetName()),
                residual);
    cv::Mat1b mask;
    cv::cvtColor(residual, mask, cv::COLOR_BGR2GRAY);
    mask = mask > 0;
    residual.setTo(0, mask == 0);

#ifdef VIDEO_TRACKING
    tracker.TrackFrame(image, &camera);

    if (cam > 0) {
      const replay::Camera& last_camera = scene.GetCamera(cam - 1);
      const cv::Mat& last_image = images.Get(last_camera.GetName());
      std::vector<Eigen::Vector2d> kp1, kp2;
      if (tracker.GetMatchingKeypoints(&last_camera, &camera, &kp1, &kp2)) {
        cv::imshow("tracks",
                   replay::VisualizeMatches(last_image, kp1, image, kp2));
        cv::waitKey(1);
      }
    }
#endif
    // Save them for later so we can estimate geometry

    sweeper.AddView(camera, residual, mask);
  }

#ifdef VIDEO_TRACKING
  std::vector<Eigen::Vector3f> reflected_points;
  std::unordered_map<int, replay::DepthMap> depth_cache;
  for (auto point : tracker.GetTracks()) {
    if (point->NumObservations() < 5) {
      continue;
    }
    point->Triangulate();
    // Eigen::Vector2d central_projection =
    // central_view->ProjectPoint(point->GetPoint());

    bool geometry_consistent = true;
    double average_reprojection_error = 0.0;
    double num_observations = static_cast<double>(point->NumObservations());
    for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
         cam += kSkipFrames) {
      const replay::Camera& camera = scene.GetCamera(cam);
      if (!point->HasObservation(&camera)) {
        continue;
      }
      if (depth_cache.count(cam) == 0) {
        depth_renderer.GetDepthMap(camera, &depth_cache[cam]);
      }

      const Eigen::Vector2d projected_coord =
          camera.ProjectPoint(point->GetPoint());
      const Eigen::Vector2d base_frame_coord =
          point->GetObservation(&camera).head<2>();
      double error = (projected_coord - base_frame_coord).norm();
      average_reprojection_error += error / num_observations;
      double tracked_depth = (point->GetPoint() - camera.GetPosition()).norm();
      double mesh_depth =
          depth_cache[cam].GetDepth(base_frame_coord.y(), base_frame_coord.x());
      if (tracked_depth / mesh_depth > 1.2 && mesh_depth > 0.0001) {
        continue;
      } else {
        geometry_consistent = false;
        break;
      }
    }
    if (geometry_consistent && average_reprojection_error < 0.5) {
      scene.AddPoint(point);
      reflected_points.emplace_back(point->GetPoint().cast<float>());
    }
  }

  replay::Plane reflected_plane(reflected_points);
  replay::Mesh reflection_mesh = reflected_plane.GetMesh();
  const int reflection_mesh_id = context->UploadMesh(reflection_mesh);

  // reflection_mesh.Save(
  // replay::JoinPath(FLAGS_output_directory, "second_layer_mesh.ply"));

  replay::Mesh pc = scene.CreatePointCloud();
  LOG(ERROR) << "Reflected point cloud has " << pc.NumVertices() << " points.";
  reflection_mesh.Save("/Users/holynski/plane.ply");

  pc.Append(scene.CreateFrustumMesh());

  pc.Save("/Users/holynski/test.ply");

#endif
  // Align the residual to find the dominant geometry
  LOG(ERROR) << "Computing depth for second layer.";
  replay::DepthMap base_depth;
  depth_renderer.GetDepthMap(*central_view, &base_depth);
  float first_layer_depth =
      base_depth.GetDepth(base_depth.Rows() / 2, base_depth.Cols() / 2);

#ifdef PLANE_SWEEP
  replay::PlaneSweepResult plane_sweep_result = sweeper.Sweep(
      *central_view, first_layer_depth * 2.0, first_layer_depth * 6.0, 200);

  cv::Mat1f reflected_depth(min_composite.size(), -1);
  cv::Mat1f smallest_variance(min_composite.size(), FLT_MAX);
  cv::Mat1b mean(min_composite.size(), 0);

  double min_variance = DBL_MAX;
  double max_variance = 0;

  for (const auto& cost_layer : plane_sweep_result.cost_volume) {
    cv::Mat1f variance = cost_layer.second.clone();
    double min;
    double max;
    cv::minMaxLoc(variance, &min, &max);
    min_variance = std::min(min, min_variance);
    max_variance = std::max(max, max_variance);
    variance.setTo(FLT_MAX,
                   plane_sweep_result.num_samples[cost_layer.first] < 3);
    // variance.setTo(FLT_MAX,
    // plane_sweep_result.num_samples[cost_layer.first] < 10);
    // cv::imwrite(
    // replay::JoinPath(
    // FLAGS_output_directory,
    //"cost_" + std::to_string(cost_layer.first / first_layer_depth) +
    //".png"),
    // cost_layer.second * 255.0 / max_cost);
    for (int row = 0; row < min_composite.rows; row++) {
      for (int col = 0; col < min_composite.cols; col++) {
        if (smallest_variance(row, col) > variance(row, col)) {
          reflected_depth(row, col) = cost_layer.first;
          smallest_variance(row, col) = variance(row, col);
          mean(row, col) = cv::norm(
              plane_sweep_result.mean_images[cost_layer.first](row, col));
        }
      }
    }
  }

  for (const auto& cost_layer : plane_sweep_result.cost_volume) {
    cv::imwrite(
        replay::JoinPath(
            FLAGS_output_directory,
            "cost_volume_" + std::to_string(cost_layer.first) + ".png"),
        replay::FloatToColor(cost_layer.second, min_variance, max_variance));
  }

  double min, max;
  reflected_depth.setTo(0, reflected_depth > first_layer_depth * 9);
  reflected_depth.setTo(0, reflected_depth <= 0);
  reflected_depth.setTo(0, mean <= 20);
  reflected_depth.setTo(0, smallest_variance > std::pow(50, 2));
  cv::minMaxLoc(reflected_depth, &min, &max);
  cv::Mat3b colormap_depth = replay::FloatToColor(reflected_depth);
  colormap_depth.setTo(0, reflected_depth == 0);
  // reflected_depth.setTo(0, smallest_variance > std::pow(10, 2));
  cv::minMaxLoc(smallest_variance, &min, &max);
  smallest_variance.setTo(max, reflected_depth > first_layer_depth * 9);
  smallest_variance.setTo(max, reflected_depth <= 0);
  smallest_variance.setTo(max, mean <= 20);
  smallest_variance.setTo(max, smallest_variance > std::pow(50, 2));
  cv::waitKey();

  std::vector<Eigen::Vector3f> reflected_points;

  for (int row = 0; row < reflected_depth.rows; row++) {
    for (int col = 0; col < reflected_depth.cols; col++) {
      if (reflected_depth(row, col) > 0.0) {
        Eigen::Vector3d point3d = central_view->UnprojectPoint(
            Eigen::Vector2d(col, row), reflected_depth(row, col));
        reflected_points.emplace_back(point3d.cast<float>());
      }
    }
  }

  replay::Plane reflected_plane(reflected_points);
  replay::Mesh reflection_mesh = reflected_plane.GetMesh();
  const int reflection_mesh_id = context->UploadMesh(reflection_mesh);
#endif

#define SINGLE_PLANE
#ifdef SINGLE_PLANE
  const Eigen::Vector3f plane_normal =
      central_view->GetLookAt().cast<float>().normalized();
  const Eigen::Vector3f plane_center =
      (central_view->GetPosition().cast<float>() +
       plane_normal * first_layer_depth * 3);

  replay::Plane reflected_plane(plane_center, -plane_normal);
  replay::Mesh reflection_mesh = reflected_plane.GetMesh(plane_center, 100000);
  const int reflection_mesh_id = context->UploadMesh(reflection_mesh);
#endif

  reflection_mesh.Save(
      replay::JoinPath(FLAGS_output_directory, "reflected.ply"));
  cv::Mat3b max_composite(image_size.y(), image_size.x(), cv::Vec3b(0, 0, 0));
  greedy_flow.SetWindowSize(50);

  for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
       cam += kSkipFrames) {
    replay::PrintProgress(
        cam + 1, std::min(scene.NumCameras(), kMaxFrames), "Computing flows",
        "- " + std::to_string(static_cast<int>(timer.ElapsedTime())) +
            " ms / frame");
    const replay::Camera& camera = scene.GetCamera(cam);
    context->BindMesh(reflection_mesh_id);
    cv::Mat2f flow_from_layer2 =
        flow_from_geometry.Calculate(*central_view, camera);
    flows_from_layer2[cam] = flow_from_layer2;
    cv::Mat2f flow_to_layer2 =
        flow_from_geometry.Calculate(camera, *central_view);
    flows_to_layer2[cam] = flow_to_layer2;
  }

  replay::ImageStackAnalyzer second_layer_statistics(options);
  cv::Mat3b max_composite_refined(image_size.y(), image_size.x(),
                                  cv::Vec3b(0, 0, 0));
  for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
       cam += kSkipFrames) {
    replay::PrintProgress(
        cam + 1, std::min(scene.NumCameras(), kMaxFrames),
        "Computing max composite",
        "- " + std::to_string(static_cast<int>(timer.ElapsedTime())) +
            " ms / frame");
    const replay::Camera& camera = scene.GetCamera(cam);
    cv::Mat3b residual_perview =
        cv::imread(replay::JoinPath(FLAGS_output_directory,

                                    "per_view_residual_0_" + camera.GetName()));
    context->BindMesh(reflection_mesh_id);

    cv::Mat3b refined_reprojected = replay::OpticalFlowAligner::InverseWarp(
        residual_perview, flows_from_layer2[cam]);

    cv::imwrite(replay::JoinPath(FLAGS_output_directory,

                                 "refined_second_plane_" + camera.GetName()),
                refined_reprojected);

    cv::Mat1b unknown_mask_refined;
    cv::cvtColor(refined_reprojected, unknown_mask_refined, cv::COLOR_BGR2GRAY);
    unknown_mask_refined = (unknown_mask_refined == 0);

    max_composite_refined.copyTo(refined_reprojected, unknown_mask_refined);
    max_composite_refined = cv::max(refined_reprojected, max_composite_refined);
    second_layer_statistics.AddImage(refined_reprojected,
                                     unknown_mask_refined == 0);

    // cv::Mat1b unknown_mask;
    // cv::cvtColor(reprojected, unknown_mask, cv::COLOR_BGR2GRAY);
    // unknown_mask = (unknown_mask == 0);
    // cv::dilate(unknown_mask, unknown_mask, cv::Mat());
    // cv::dilate(unknown_mask, unknown_mask, cv::Mat());
    // reprojected.setTo(cv::Vec3b(0, 0, 0), unknown_mask);

    // max_composite.copyTo(reprojected, unknown_mask);
    // max_composite = cv::max(reprojected, max_composite);
  }

  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "max_composite.png"),
              max_composite_refined);

  max_composite = max_composite_refined;

  LOG(ERROR) << "Refining layers...";

  // central_view->SetImageSize(Eigen::Vector2i(1920 / 8, 1080 / 16));
  // cv::resize(min_composite, min_composite, cv::Size(1920 / 8, 1080 / 16));
  // cv::resize(max_composite, max_composite, cv::Size(1920 / 8, 1080 / 16));

  // replay::LayerRefiner refiner(context, *central_view, mesh,
  // plane_sweep_result.meshes[lowest_mean_index]);
  cv::Mat1b mask(max_composite.size(), 0.0);
  static const int kNumLayer2MotionIterations = 8;
  for (int it = kNumLayer2MotionIterations; it >= 1; it++) {
    const float kDownscaleFactor = std::pow(2, it);
    LOG(ERROR) << "Refinement iteration " << it;
    replay::ReflectionSegmenter segmenter(context, *central_view, min_composite,
                                          max_composite, mesh, reflection_mesh);
    for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
         cam += kSkipFrames) {
      LOG(ERROR) << ">>Adding image " << cam << "/" << scene.NumCameras();
      const replay::Camera& camera = scene.GetCamera(cam);

      cv::Mat image = images.Get(camera.GetName()).clone();
      replay::ExposureAlignment::TransformImageExposure(
          image, camera.GetExposure(), Eigen::Vector3f(1, 1, 1), &image);

      segmenter.AddImage(image, camera);
    }

    LOG(ERROR) << "Optimizing...";

    cv::Mat3b partition_image = cv::imread(FLAGS_partition_image);
    LOG(ERROR) << "Partition image: "
               << (partition_image.empty() ? "NOT FOUND" : "FOUND");
    cv::Mat1b partitions(min_composite.size(), 255);

    if (!partition_image.empty()) {
      for (int row = 0; row < partitions.rows; row++) {
        for (int col = 0; col < partitions.cols; col++) {
          if (partition_image(row, col) == cv::Vec3b(0, 0, 255)) {
            partitions(row, col) = 255;
          } else {
            partitions(row, col) = 0;
          }
        }
      }
      cv::imwrite(replay::JoinPath(FLAGS_output_directory, "partitions.png"),
                  partitions);
    }

    CHECK(segmenter.Optimize(mask, partitions));

    cv::Mat3b mask_3c;
    cv::cvtColor(mask, mask_3c, cv::COLOR_GRAY2BGR);

    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "mask_" + std::to_string(it) + ".png"),
                mask);

    replay::LayerRefiner refiner(max_composite.cols, max_composite.rows);
    replay::ImageReprojector image_reprojector3(context);
    cv::destroyAllWindows();
    for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
         cam += kSkipFrames) {
      const replay::Camera& camera = scene.GetCamera(cam);
      cv::Mat image = images.Get(camera.GetName()).clone();
      cv::Mat1b layer1_mask_reprojected;
      // context->BindMesh(mesh_id);
      // image_reprojector3.SetImage(mask);
      // image_reprojector3.SetSourceCamera(*central_view);
      layer1_mask_reprojected =
          replay::OpticalFlowAligner::InverseWarp(mask, flows_to_layer1[cam]);
      // image_reprojector3.Reproject(camera, &layer1_mask_reprojected, 0.25);
      cv::Mat3b warped1, warped2;
      refiner.AddImage(image, flows_to_layer1[cam], flows_to_layer2[cam],
                       layer1_mask_reprojected);
    }

    cv::Mat3b first_layer = min_composite.clone();
    refiner.Optimize(first_layer, max_composite, 5);
    first_layer.copyTo(min_composite, mask == 255);

    cv::Mat1f alpha(min_composite.size(), 0.0);
    alpha.setTo(1.0, mask > 0);
    for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
         cam += kSkipFrames) {
      const replay::Camera& camera = scene.GetCamera(cam);
      cv::Mat image = images.Get(camera.GetName()).clone();
      replay::CompositeMotionRefiner motion_refiner(image.cols, image.rows);
      // motion_refiner.Optimize(min_composite, max_composite, alpha, image,
      // flows_to_layer1[cam], flows_to_layer2[cam], 5);
    }

    for (int cam = 0; cam < std::min(scene.NumCameras(), kMaxFrames);
         cam += kSkipFrames) {
      cv::Mat3b diffuse = replay::OpticalFlowAligner::InverseWarp(
          min_composite, flows_to_layer1[cam]);
      const replay::Camera& camera = scene.GetCamera(cam);
      const cv::Mat3b image = images.Get(camera.GetName());
      const cv::Mat3b residual = image - diffuse;
      cv::Mat2f refined_flow = flows_to_layer2[cam].clone();
      greedy_flow.calc(residual, max_composite, refined_flow);
      cv::Mat3b refined_reprojected =
          replay::OpticalFlowAligner::InverseWarp(residual, refined_flow);
      flows_from_layer2[cam] = refined_flow;
      cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                   "per_view_residual_" + std::to_string(it) +
                                       "_" + camera.GetName()),
                  residual);
    }

    cv::imwrite(replay::JoinPath(
                    FLAGS_output_directory,
                    "max_composite_optimized_" + std::to_string(it) + ".png"),
                max_composite);
    cv::imwrite(replay::JoinPath(
                    FLAGS_output_directory,
                    "min_composite_optimized_" + std::to_string(it) + ".png"),
                min_composite);
  }

  replay::ImageReprojector image_reprojector4(context);
  replay::ImageReprojector image_reprojector5(context);
  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    replay::PrintProgress(
        cam + 1, scene.NumCameras(), "Recomposing layers",
        "- " + std::to_string(static_cast<int>(timer.ElapsedTime())) +
            " ms / frame");

    const replay::Camera& camera = scene.GetCamera(cam);

    // Reproject the residual images back into the source viewpoints
    context->BindMesh(mesh_id);
    cv::Mat3b layer1_min_reprojected;
    image_reprojector5.SetImage(min_composite);
    image_reprojector5.SetSourceCamera(*central_view);
    image_reprojector5.Reproject(camera, &layer1_min_reprojected, 0.25);
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "layer1_at_" + camera.GetName()),
                layer1_min_reprojected);

    cv::Mat1b layer1_mask_reprojected;
    image_reprojector5.SetImage(mask);
    image_reprojector5.SetSourceCamera(*central_view);
    image_reprojector5.Reproject(camera, &layer1_mask_reprojected, 0.25);
    cv::imwrite(
        replay::JoinPath(FLAGS_output_directory, "mask_at_" + camera.GetName()),
        layer1_mask_reprojected);

    context->BindMesh(reflection_mesh_id);
    cv::Mat3b layer2_reprojected;
    image_reprojector4.SetImage(max_composite);
    image_reprojector4.SetSourceCamera(*central_view);
    image_reprojector4.Reproject(camera, &layer2_reprojected);
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "layer2_at_" + camera.GetName()),
                layer2_reprojected);

    // cv::Mat3b composed = layer1_mean_reprojected + layer2_reprojected;

    layer2_reprojected.setTo(0, layer1_mask_reprojected == 0);
    cv::Mat3b composed(layer2_reprojected.size());
    cv::add(layer2_reprojected, layer1_min_reprojected, composed);
    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "composed_" + camera.GetName()),
                composed);
  }

  return 0;
}
