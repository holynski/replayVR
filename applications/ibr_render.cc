#include <gflags/gflags.h>
#include <glog/logging.h>
#include <replay/camera/camera.h>
#include <replay/camera/pinhole_camera.h>
#include <replay/flow/optical_flow_aligner.h>
#include <replay/image/fuzzy_difference.h>
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
  // replay::DepthCache depths(FLAGS_depth_cache, 300);
  replay::ImageCache min_composite_cache(FLAGS_min_composite_cache, 300);
  // replay::ImageCache min_composite_residual_cache(
  // FLAGS_min_composite_residual_cache, 300);
  // replay::ImageCache capreal_cache(FLAGS_capreal_cache, 300);
  // replay::ImageCache capreal_residual_cache(FLAGS_capreal_residual_cache,
  // 300);
  replay::ReadBundler(FLAGS_bundler_file, FLAGS_image_list, images, &scene);

  replay::Mesh frustum_mesh = scene.CreateFrustumMesh();
  CHECK_GT(scene.NumCameras(), 0);

  LOG(INFO) << "Loading the mesh.";
  CHECK(replay::FileExists(FLAGS_mesh));
  replay::Mesh mesh;
  CHECK(mesh.Load(FLAGS_mesh));
  CHECK(mesh.NumTriangleFaces() > 0) << "Mesh is empty";
  frustum_mesh.Append(mesh);
  frustum_mesh.Save(replay::JoinPath(FLAGS_output_directory, "frustum.ply"));

  LOG(ERROR) << "Initializing context";
  std::shared_ptr<replay::OpenGLContext> context =
      std::make_shared<replay::OpenGLContext>();
  CHECK(context->Initialize());

  replay::PlaneSweep ps(context);
  replay::ModelRenderer renderer(context);
  CHECK(renderer.Initialize());
  LOG(ERROR) << "Uploading mesh";
  int mesh_id = context->UploadMesh(mesh);
  context->BindMesh(mesh_id);

  replay::ExposureAlignment::Options exposure_options;
  // exposure_options.single_channel = true;
  replay::ExposureAlignment exposure(context, exposure_options, images, &scene);
  // TODO(holynski): Add support for non-recon images in exposure alignment
  LOG(ERROR) << "Aligning exposure";
  exposure.GenerateExposureCoefficients();

  replay::MinCompositorSequential min_compositor(context);
  context->BindMesh(mesh_id);
  LOG(ERROR) << "Uploading min composite images";
  CHECK(min_compositor.Initialize());

  int k = 0;
  int start_index = 0;
  int end_index = scene.NumCameras();

  replay::OpticalFlowAligner aligner(replay::OpticalFlowType::TVL1);
  replay::FuzzyMinDifference<cv::Vec3b> min_difference(context);

  replay::DepthMapRenderer depth_renderer(context);
  depth_renderer.Initialize();
  replay::Mesh window_mesh;
  int window_mesh_id = -1;
  if (replay::FileExists(FLAGS_window_mesh)) {
    CHECK(window_mesh.Load(FLAGS_window_mesh));
    window_mesh_id = context->UploadMesh(window_mesh);
  }

  double max_time = 0;
  LOG(ERROR) << "Starting loop";
  for (int i = start_index; i < end_index; i++) {
    replay::SimpleTimer timer;
    // Set the interpolated position.
    cv::Mat3b output;
    const replay::Camera& camera = scene.GetCamera(i);
    cv::Mat image = images.Get(camera.GetName()).clone();
    replay::ExposureAlignment::TransformImageExposure(
        image, camera.GetExposure(), Eigen::Vector3f(1, 1, 1), &image);
    context->BindMesh(mesh_id);
    LOG(ERROR) << "Rendering diffuse";
    // output = capreal_cache.Get("/diffuse_" + camera.GetName());
    // if (output.empty()) {
    CHECK(renderer.RenderView(scene.GetCamera(i), &output));
    //}
    // output = aligner.Align(output, image);

    cv::imwrite(FLAGS_output_directory + "/diffuse_" + camera.GetName(),
                output);
    cv::imwrite(FLAGS_output_directory + "/orig_" + camera.GetName(), image);
    cv::Mat3b minc;
    LOG(ERROR) << "Getting min composite";
    minc = min_composite_cache.Get("/minc_" + camera.GetName());
    if (minc.empty()) {
      CHECK(min_compositor.GetMinComposite(scene, images, scene.GetCamera(i),
                                           &minc));
    }
    LOG(ERROR) << "Got min composite";
    // minc = aligner.Align(minc, image);
    cv::imwrite(FLAGS_output_directory + "/minc_" + camera.GetName(), minc);
    cv::Mat3f ift, oft;
    cv::Mat3f mft;
    image.convertTo(ift, CV_32FC3);
    output.convertTo(oft, CV_32FC3);
    minc.convertTo(mft, CV_32FC3);
    cv::pow(ift, 2.2, ift);
    cv::pow(oft, 2.2, oft);
    cv::pow(mft, 2.2, mft);
    cv::Mat3b specular = min_difference.GetDifference(image, output, 10);
    cv::Mat3b specular_m = min_difference.GetDifference(image, minc, 10);
    cv::imwrite(FLAGS_output_directory + "/specular_" + camera.GetName(),
                specular);
    cv::imwrite(FLAGS_output_directory + "/res_minc_" + camera.GetName(),
                specular_m);

    cv::Mat global;
    cv::Mat1b valid_reflections;
    if (window_mesh_id >= 0) {
      replay::DepthMap map;
      context->BindMesh(window_mesh_id);
      depth_renderer.GetDepthMap(camera, &map);
      context->BindMesh(mesh_id);
      // map.WriteDepthAsRGB("output/test.png");
      valid_reflections = map.Depth() > 0;
      specular_m.setTo(0, valid_reflections == 0);
    }

    k++;
    max_time += timer.ElapsedTime();
    LOG(ERROR) << "Processed frame " << k << "/" << scene.NumCameras()
               << "). Average time per frame: " << max_time / k << "ms";
  }

  const replay::Camera& center_view = scene.GetCamera(scene.NumCameras() / 2);
  const replay::Camera& center_view2 =
      scene.GetCamera(scene.NumCameras() / 2 + 5);
  float window_depth = 0;
  cv::Mat1b valid_pixels1(center_view.GetImageSize()[1],
                          center_view.GetImageSize()[0], 255);
  cv::Mat1b valid_pixels1(center_view2.GetImageSize()[1],
                          center_view2.GetImageSize()[0], 255);
  if (window_mesh_id >= 0) {
    replay::DepthMap map;
    context->BindMesh(window_mesh_id);
    depth_renderer.GetDepthMap(center_view, &map);
    window_depth = cv::mean(map.Depth(), map.Depth() > 0)[0];
    valid_pixels1 = map.Depth() > 0;
    depth_renderer.GetDepthMap(center_view2, &map);
    valid_pixels2 = map.Depth() > 0;
  } else {
//add support for no windows
//so we can look at the belgrave results without the masks
  }

  replay::Camera* maxc_view = center_view.Clone();
  Eigen::Vector2d fov = maxc_view->GetFOV();
  LOG(ERROR) << "FOV: " << fov;
  maxc_view->SetFocalLengthFromFOV(Eigen::Vector2d(fov.x() * 2, fov.y()));
  maxc_view->SetImageSize(Eigen::Vector2i(1920 * 2, 1080));

  cv::Mat3b img1 =
      cv::imread(FLAGS_output_directory + "/res_minc_" + center_view.GetName());
  cv::Mat3b img2 = cv::imread(FLAGS_output_directory + "/res_minc_" +
                              center_view2.GetName());
  LOG(ERROR) << "Window depth: " << window_depth;
  auto volume =
      ps.Sweep(center_view, center_view2, img1, valid_pixels1, img2,
               valid_pixels2, window_depth * 2, window_depth * 10, 100);
  double minval, maxval;
  cv::minMaxLoc(volume.begin()->second, &minval, &maxval);
  float best = -1;
  float best_mean = 999999;
  for (auto plane : volume) {
    float mean = cv::mean(plane.second, plane.second > 0.0)[0];
    if (mean < best_mean) {
      best_mean = mean;
      best = plane.first;
    }
    cv::imwrite(FLAGS_output_directory + "/sweep_" +
                    std::to_string(plane.first) + ".png",
                plane.second* 255.0 /= maxval);
  }
  cv::imwrite(
      FLAGS_output_directory + "/sweep_best_" + std::to_string(best) + ".png",
      volume[best]* 255.0 /= maxval);

  Eigen::Vector3f plane_center = center_view.GetPosition().cast<float>() +
                                 center_view.GetLookAt().cast<float>() * best;
  replay::Mesh reflection_proxy = replay::Mesh::Plane(
      plane_center, center_view.GetLookAt().cast<float>().normalized(),
      Eigen::Vector2f(best * 10, best * 10));
  reflection_proxy.Save(
      replay::JoinPath(FLAGS_output_directory, "reflection.ply"));

  int reflection_mesh_id = context->UploadMesh(reflection_proxy);
  context->BindMesh(reflection_mesh_id);

  replay::ImageReprojector image_reprojector(context);
  cv::Mat3b max_reflection(maxc_view->GetImageSize().y(),
                           maxc_view->GetImageSize().x(), cv::Vec3b(0, 0, 0));

  k = 0;
  for (int i = start_index; i < end_index; i++) {
    cv::Mat1b valid_reflections;
    const replay::Camera& camera = scene.GetCamera(i);
    if (window_mesh_id >= 0) {
      replay::DepthMap map;
      context->BindMesh(window_mesh_id);
      depth_renderer.GetDepthMap(camera, &map);
      valid_reflections = map.Depth() > 0;
      cv::Mat3b reprojected_reflection;

      cv::Mat3b reflection =
          cv::imread(FLAGS_output_directory + "/res_minc_" + camera.GetName());
      if (!valid_reflections.empty()) {
        reflection.setTo(0, valid_reflections == 0);
      }
      context->BindMesh(reflection_mesh_id);
      image_reprojector.SetImage(reflection);
      image_reprojector.SetSourceCamera(camera);
      image_reprojector.Reproject(*maxc_view, &reprojected_reflection);

      cv::imwrite(FLAGS_output_directory + "/global_ref_" + camera.GetName(),
                  reprojected_reflection);

      max_reflection = cv::max(max_reflection, reprojected_reflection);
    }
    k++;
  }
  k = 0;
  for (int i = start_index; i < end_index; i++) {
    const replay::Camera& camera = scene.GetCamera(i);
    cv::Mat1b valid_reflections;
    cv::Mat3b reprojected_reflection;
    context->BindMesh(reflection_mesh_id);
    image_reprojector.SetImage(max_reflection);
    image_reprojector.SetSourceCamera(*maxc_view);
    image_reprojector.Reproject(camera, &reprojected_reflection);
    if (window_mesh_id >= 0) {
      replay::DepthMap map;
      context->BindMesh(window_mesh_id);
      depth_renderer.GetDepthMap(camera, &map);
      valid_reflections = map.Depth() > 0;
      if (!valid_reflections.empty()) {
        reprojected_reflection.setTo(0, valid_reflections == 0);
      }
      cv::Mat3b diffuse =
          cv::imread(FLAGS_output_directory + "/diffuse_" + camera.GetName());
      cv::Mat3b minc =
          cv::imread(FLAGS_output_directory + "/minc_" + camera.GetName());
      cv::imwrite(FLAGS_output_directory + "/mcomposed_" + camera.GetName(),
                  reprojected_reflection + minc);
      cv::imwrite(FLAGS_output_directory + "/composed_" + camera.GetName(),
                  reprojected_reflection + diffuse);
    }
  }
  cv::imwrite(FLAGS_output_directory + "/max_reflection.png", max_reflection);
}
