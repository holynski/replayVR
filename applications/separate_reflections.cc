#include <gflags/gflags.h>
#include <glog/logging.h>
#include <replay/camera/camera.h>
#include <replay/camera/pinhole_camera.h>
#include <replay/flow/optical_flow_aligner.h>
#include <replay/image/fuzzy_difference.h>
#include <replay/image/image_stack_analyzer.h>
#include <replay/image/sum_absolute_difference.h>
#include <replay/io/read_bundler.h>
#include <replay/io/read_capreal.h>
#include <replay/multiview/exposure_alignment.h>
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

#include <Eigen/Geometry>

DEFINE_string(bundler_file, "", "");
DEFINE_string(image_list, "", "");
DEFINE_string(images_directory, "", "");
DEFINE_string(output_directory, "", "");
DEFINE_string(mesh, "", "");
DEFINE_string(window_mesh, "", "");
DEFINE_string(min_composite_cache, "", "");
DEFINE_string(depth_cache, "", "");
// DEFINE_string(capreal_cache, "", "");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  replay::Reconstruction scene;
  CHECK(replay::FileExists(FLAGS_bundler_file))
      << "Reconstruction file (" << FLAGS_bundler_file << ") doesn't exist!";
  CHECK(replay::FileExists(FLAGS_image_list))
      << "Reconstruction file (" << FLAGS_image_list << ") doesn't exist!";

  // Initialize caches. These are where the images are stored, so they don't
  // need to be recomputed each time.
  replay::ImageCache images(FLAGS_images_directory, 300);
  replay::ImageCache min_composite_cache(FLAGS_min_composite_cache, 300);
  replay::ReadBundler(FLAGS_bundler_file, FLAGS_image_list, images, &scene);

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

  replay::PlaneSweep ps(context);
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

  replay::ExposureAlignment::Options exposure_options;
  replay::ExposureAlignment exposure(context, exposure_options, images, &scene);
  LOG(ERROR) << "Aligning exposure";
  exposure.GenerateExposureCoefficients();

  /*
   * Render the first layer images
   */

  // Render the diffuse mesh once from the central viewpoint.
  // cv::Mat3b textured_mesh;
  context->BindMesh(mesh_id);
  // CHECK(renderer.RenderView(*central_view, &textured_mesh));
  // cv::imwrite(replay::JoinPath(FLAGS_output_directory, "textured.png"),
  // textured_mesh);

  replay::ImageStackAnalyzer first_layer_statistics;
  replay::SimpleTimer timer;
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

    cv::imwrite(replay::JoinPath(FLAGS_output_directory,
                                 "reprojected_" + camera.GetName()),
                reprojected);

    first_layer_statistics.AddImage(reprojected, unknown_mask == 0);
  }

  cv::Mat3b min_composite = first_layer_statistics.GetMin();
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "min_composite.png"),
              min_composite);

  cv::Mat3b mean = first_layer_statistics.GetMean();

  replay::ImageStackAnalyzer first_layer_aligned_statistics;
  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    LOG(ERROR) << "Computing layer 1, frame (" << cam << "/"
               << scene.NumCameras() << "): " << timer.ElapsedTime() << "ms";
    const replay::Camera& camera = scene.GetCamera(cam);
    cv::Mat3b image = cv::imread(replay::JoinPath(
        FLAGS_output_directory, "reprojected_" + camera.GetName()));
    cv::Mat3b aligned;
    aligned = aligner.Align(image, mean);
    cv::Mat1b unknown_mask;
    cv::cvtColor(aligned, unknown_mask, cv::COLOR_BGR2GRAY);
    unknown_mask = unknown_mask == 0;
    cv::dilate(unknown_mask, unknown_mask, cv::Mat());
    cv::imwrite(
        replay::JoinPath(FLAGS_output_directory, "flow_" + camera.GetName()),
        aligned);

    first_layer_aligned_statistics.AddImage(aligned, unknown_mask == 0);
  }
  cv::Mat3f variance_aligned_rgb = first_layer_aligned_statistics.GetVariance();
  cv::Mat1f variance_aligned;
  cv::cvtColor(variance_aligned_rgb, variance_aligned, cv::COLOR_BGR2GRAY);

  cv::Mat3f variance_rgb = first_layer_statistics.GetVariance();
  cv::Mat1f variance;
  cv::cvtColor(variance_rgb, variance, cv::COLOR_BGR2GRAY);
  cv::Mat1b mean_gray;
  cv::cvtColor(mean, mean_gray, cv::COLOR_BGR2GRAY);
  cv::Mat1f mean_gradient;
  cv::Laplacian(mean_gray, mean_gradient, CV_32F, 15);
  double min, max;
  cv::minMaxLoc(variance, &min, &max);
  LOG(ERROR) << "Variance min/max: " << min << "/" << max;
  cv::Mat1f scaled_variance;
  mean_gradient = cv::abs(mean_gradient);
  cv::minMaxLoc(mean_gradient, &min, &max);
  LOG(ERROR) << "Laplacian min/max: " << min << "/" << max;
  mean_gradient /= 10000000;
  // mean_gradient *= ;
  cv::divide(variance, mean_gradient + 1, scaled_variance);
  cv::minMaxLoc(scaled_variance, &min, &max);
  LOG(ERROR) << "Scaled variance min/max: " << min << "/" << max;

  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "mean.png"), mean);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "variance.png"),
              variance / 10);
   cv::imwrite(replay::JoinPath(FLAGS_output_directory,
   "variance_aligned.png"), variance_aligned / 10);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "scaled_variance.png"),
              scaled_variance / 10);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "mean_gradient.png"),
              mean_gradient * 10);

  float variance_threshold = 20;
  cv::Mat1b variance_mask = scaled_variance > variance_threshold;
  cv::erode(variance_mask, variance_mask, cv::Mat());
  cv::erode(variance_mask, variance_mask, cv::Mat());
  cv::erode(variance_mask, variance_mask, cv::Mat());
  cv::dilate(variance_mask, variance_mask, cv::Mat());
  cv::dilate(variance_mask, variance_mask, cv::Mat());
  cv::dilate(variance_mask, variance_mask, cv::Mat());
  // cv::imshow("variance_mask", variance_mask);
  // cv::waitKey();

  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "first_layer_mask.png"),
              variance_mask);

  replay::PlaneSweep sweeper(context);
  replay::ImageReprojector image_reprojector2(context);
  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    LOG(ERROR) << "Computing layer 1 residual, frame (" << cam << "/"
               << scene.NumCameras() << "): " << timer.ElapsedTime() << "ms";
    const replay::Camera& camera = scene.GetCamera(cam);

    cv::Mat3b image = cv::imread(replay::JoinPath(
        FLAGS_output_directory, "reprojected_" + camera.GetName()));
    cv::Mat3b residual = min_difference.GetDifference(image, min_composite, 10);
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
      *central_view, first_layer_depth * 2.0, first_layer_depth * 4.0, 10);

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
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "second_layer_min.png"),
              plane_sweep_result.min_images[lowest_mean_index]);
  cv::imwrite(replay::JoinPath(FLAGS_output_directory, "second_layer_max.png"),
              plane_sweep_result.max_images[lowest_mean_index]);

  LOG(ERROR) << "Chose layer: " << lowest_mean_index / first_layer_depth;

  cv::Mat3b layer1_mean =
      cv::imread(replay::JoinPath(FLAGS_output_directory, "mean.png"));
  cv::Mat3b layer1_min =
      cv::imread(replay::JoinPath(FLAGS_output_directory, "min_composite.png"));
  cv::Mat3b layer1_mask = cv::imread(
      replay::JoinPath(FLAGS_output_directory, "first_layer_mask.png"));
  cv::Mat3b layer2 = cv::imread(
      replay::JoinPath(FLAGS_output_directory, "second_layer_max.png"));
  replay::ImageReprojector image_reprojector3(context);
  replay::ImageReprojector image_reprojector4(context);
  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    LOG(ERROR) << "Recomposing layer 1 + 2 (" << cam << "/"
               << scene.NumCameras() << "): " << timer.ElapsedTime() << "ms";
    const replay::Camera& camera = scene.GetCamera(cam);

    // Reproject the residual images back into the source viewpoints
    context->BindMesh(mesh_id);
    cv::Mat3b layer1_mean_reprojected;
    image_reprojector3.SetImage(layer1_mean);
    image_reprojector3.SetSourceCamera(*central_view);
    image_reprojector3.Reproject(camera, &layer1_mean_reprojected, 0.25);

    cv::Mat3b layer1_min_reprojected;
    image_reprojector3.SetImage(layer1_min);
    image_reprojector3.SetSourceCamera(*central_view);
    image_reprojector3.Reproject(camera, &layer1_min_reprojected);

    cv::Mat3b layer1_mask_reprojected;
    image_reprojector3.SetImage(layer1_mask);
    image_reprojector3.SetSourceCamera(*central_view);
    image_reprojector3.Reproject(camera, &layer1_mask_reprojected);
    cv::Mat1b specular_mask;
    cv::cvtColor(layer1_mask_reprojected, specular_mask, cv::COLOR_BGR2GRAY);

    context->BindMesh(plane_sweep_result.mesh_ids[lowest_mean_index]);
    cv::Mat3b layer2_reprojected;
    image_reprojector3.SetImage(layer2);
    image_reprojector3.SetSourceCamera(*central_view);
    image_reprojector3.Reproject(camera, &layer2_reprojected);

    // cv::Mat3b residual = cv::imread(replay::JoinPath(
    // FLAGS_output_directory, "per_view_residual_" + camera.GetName()));
    // cv::Mat3b residual_central;
    // context->BindMesh(plane_sweep_result.mesh_ids[lowest_mean_index]);
    // image_reprojector4.SetImage(residual);
    // image_reprojector4.SetSourceCamera(camera);
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

  // Reproject all the residual images to the central viewpoint using the
  // aligned geometry

  // Determine the per-pixel variance

  //
}
// LOG(ERROR) << "Starting loop";
// for (int i = start_index; i < end_index; i++) {
//// Set the interpolated position.
// cv::Mat3b output;
// const replay::Camera& camera = scene.GetCamera(i);
// cv::Mat image = images.Get(camera.GetName()).clone();
// replay::ExposureAlignment::TransformImageExposure(
// image, camera.GetExposure(), Eigen::Vector3f(1, 1, 1), &image);
// context->BindMesh(mesh_id);
// LOG(ERROR) << "Rendering diffuse";
//// output = capreal_cache.Get("/diffuse_" + camera.GetName());
//// if (output.empty()) {
// CHECK(renderer.RenderView(scene.GetCamera(i), &output));
////}
//// output = aligner.Align(output, image);

// cv::imwrite(FLAGS_output_directory + "/diffuse_" + camera.GetName(),
// output);
// cv::imwrite(FLAGS_output_directory + "/orig_" + camera.GetName(), image);
// cv::Mat3b minc;
// LOG(ERROR) << "Getting min composite";
// minc = min_composite_cache.Get("/minc_" + camera.GetName());
// if (minc.empty()) {
// CHECK(min_compositor.GetMinComposite(scene, images, scene.GetCamera(i),
//&minc));
//}
// LOG(ERROR) << "Got min composite";
//// minc = aligner.Align(minc, image);
// cv::imwrite(FLAGS_output_directory + "/minc_" + camera.GetName(), minc);
// cv::Mat3f ift, oft;
// cv::Mat3f mft;
// image.convertTo(ift, CV_32FC3);
// output.convertTo(oft, CV_32FC3);
// minc.convertTo(mft, CV_32FC3);
// cv::pow(ift, 2.2, ift);
// cv::pow(oft, 2.2, oft);
// cv::pow(mft, 2.2, mft);
// cv::Mat3b specular = min_difference.GetDifference(image, output, 10);
// cv::Mat3b specular_m = min_difference.GetDifference(image, minc, 10);
// cv::imwrite(FLAGS_output_directory + "/specular_" + camera.GetName(),
// specular);
// cv::imwrite(FLAGS_output_directory + "/res_minc_" + camera.GetName(),
// specular_m);

// cv::Mat global;
// cv::Mat1b valid_reflections;
// if (window_mesh_id >= 0) {
// replay::DepthMap map;
// context->BindMesh(window_mesh_id);
// depth_renderer.GetDepthMap(camera, &map);
// context->BindMesh(mesh_id);
//// map.WriteDepthAsRGB("output/test.png");
// valid_reflections = map.Depth() > 0;
// specular_m.setTo(0, valid_reflections == 0);
//}

// k++;
//}

// const replay::Camera& center_view2 =
// scene.GetCamera(scene.NumCameras() / 2 + 5);
// float window_depth = 0;
// cv::Mat1b valid_pixels1(center_view.GetImageSize()[1],
// center_view.GetImageSize()[0], 255);
// cv::Mat1b valid_pixels2(center_view2.GetImageSize()[1],
// center_view2.GetImageSize()[0], 255);
// if (window_mesh_id >= 0) {
// replay::DepthMap map;
// context->BindMesh(window_mesh_id);
// depth_renderer.GetDepthMap(center_view, &map);
// window_depth = cv::mean(map.Depth(), map.Depth() > 0)[0];
// valid_pixels1 = map.Depth() > 0;
// depth_renderer.GetDepthMap(center_view2, &map);
// valid_pixels2 = map.Depth() > 0;
//} else {
// replay::DepthMap map;
// context->BindMesh(mesh_id);
// depth_renderer.GetDepthMap(center_view, &map);
// window_depth = cv::mean(map.Depth(), map.Depth() > 0)[0];
//}

// replay::Camera* maxc_view = center_view.Clone();
// Eigen::Vector2d fov = maxc_view->GetFOV();
// LOG(ERROR) << "FOV: " << fov;
// maxc_view->SetFocalLengthFromFOV(Eigen::Vector2d(fov.x() * 2, fov.y()));
// maxc_view->SetImageSize(Eigen::Vector2i(1920 * 2, 1080));

// cv::Mat3b img1 =
// cv::imread(FLAGS_output_directory + "/res_minc_" + center_view.GetName());
// cv::Mat3b img2 = cv::imread(FLAGS_output_directory + "/res_minc_" +
// center_view2.GetName());
// LOG(ERROR) << "Window depth: " << window_depth;
// auto volume =
// ps.Sweep(center_view, center_view2, img1, valid_pixels1, img2,
// valid_pixels2, window_depth * 2, window_depth * 10, 100);
// double minval, maxval;
// cv::minMaxLoc(volume.begin()->second, &minval, &maxval);
// float best = -1;
// float best_mean = 999999;
// for (auto plane : volume) {
// float mean = cv::mean(plane.second, plane.second > 0.0)[0];
// if (mean < best_mean) {
// best_mean = mean;
// best = plane.first;
//}
// cv::imwrite(FLAGS_output_directory + "/sweep_" +
// std::to_string(plane.first) + ".png",
// plane.second* 255.0 /= maxval);
//}
// cv::imwrite(
// FLAGS_output_directory + "/sweep_best_" + std::to_string(best) + ".png",
// volume[best]* 255.0 /= maxval);

// Eigen::Vector3f plane_center = center_view.GetPosition().cast<float>() +
// center_view.GetLookAt().cast<float>() * best;
// replay::Mesh reflection_proxy = replay::Mesh::Plane(
// plane_center, center_view.GetLookAt().cast<float>().normalized(),
// Eigen::Vector2f(best * 10, best * 10));
// reflection_proxy.Save(
// replay::JoinPath(FLAGS_output_directory, "reflection.ply"));

// int reflection_mesh_id = context->UploadMesh(reflection_proxy);
// context->BindMesh(reflection_mesh_id);

// cv::Mat3b max_reflection(maxc_view->GetImageSize().y(),
// maxc_view->GetImageSize().x(), cv::Vec3b(0, 0, 0));

// k = 0;
// replay::SumAbsoluteDifference<cv::Vec3b> sad(context);
// for (int i = start_index; i < end_index; i++) {
// cv::Mat1b valid_reflections;
// const replay::Camera& camera = scene.GetCamera(i);
// if (window_mesh_id >= 0) {
// replay::DepthMap map;
// context->BindMesh(window_mesh_id);
// depth_renderer.GetDepthMap(camera, &map);
// valid_reflections = map.Depth() > 0;
//}
// cv::Mat3b reprojected_reflection;

// cv::Mat3b reflection =
// cv::imread(FLAGS_output_directory + "/res_minc_" + camera.GetName());
// if (!valid_reflections.empty()) {
// reflection.setTo(0, valid_reflections == 0);
//}
// context->BindMesh(reflection_mesh_id);
// image_reprojector.SetImage(reflection);
// image_reprojector.SetSourceCamera(camera);
// image_reprojector.Reproject(*maxc_view, &reprojected_reflection);
// cv::Mat1b valid_reprojected;
// cv::inRange(reprojected_reflection, cv::Scalar(1, 1, 1),
// cv::Scalar(255, 255, 255), valid_reprojected);
// cv::Mat1b valid_max;
// cv::inRange(max_reflection, cv::Scalar(1, 1, 1), cv::Scalar(255, 255, 255),
// valid_max);
// cv::Mat1f cost = sad.GetDifference(reprojected_reflection, max_reflection,
// valid_reprojected, valid_max);
// cost.setTo(0, cost > 900);
// cv::imshow("cost", cost);
// cv::imshow("thresh", cost < 0.1);

// cv::imwrite(FLAGS_output_directory + "/global_ref_" + camera.GetName(),
// reprojected_reflection);
// cv::imshow("repro", reprojected_reflection);
// cv::Mat3b masked(reprojected_reflection.rows, reprojected_reflection.cols,
// cv::Vec3b(0, 0, 0));
// reprojected_reflection.copyTo(masked, cost < 0.1);
// cv::imshow("repro_masked", masked);
// max_reflection = cv::max(max_reflection, masked);
// cv::imshow("max", max_reflection);
// cv::waitKey();
// k++;
//}
// k = 0;
// for (int i = start_index; i < end_index; i++) {
// const replay::Camera& camera = scene.GetCamera(i);
// cv::Mat1b valid_reflections;
// cv::Mat3b reprojected_reflection;
// context->BindMesh(reflection_mesh_id);
// image_reprojector.SetImage(max_reflection);
// image_reprojector.SetSourceCamera(*maxc_view);
// image_reprojector.Reproject(camera, &reprojected_reflection);
// if (window_mesh_id >= 0) {
// replay::DepthMap map;
// context->BindMesh(window_mesh_id);
// depth_renderer.GetDepthMap(camera, &map);
// valid_reflections = map.Depth() > 0;
//}
// if (!valid_reflections.empty()) {
// reprojected_reflection.setTo(0, valid_reflections == 0);
//}
// cv::Mat3b diffuse =
// cv::imread(FLAGS_output_directory + "/diffuse_" + camera.GetName());
// cv::Mat3b minc =
// cv::imread(FLAGS_output_directory + "/minc_" + camera.GetName());
// cv::imwrite(FLAGS_output_directory + "/mcomposed_" + camera.GetName(),
// reprojected_reflection + minc);
// cv::imwrite(FLAGS_output_directory + "/composed_" + camera.GetName(),
// reprojected_reflection + diffuse);
//}
// cv::imwrite(FLAGS_output_directory + "/max_reflection.png", max_reflection);
//}
