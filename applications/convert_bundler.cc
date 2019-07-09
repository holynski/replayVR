#include <gflags/gflags.h>
#include <glog/logging.h>
#include <replay/io/read_bundler.h>
#include <replay/geometry/mesh.h>
#include <replay/multiview/exposure_alignment.h>
#include <replay/rendering/opengl_context.h>
#include <replay/sfm/reconstruction.h>
#include <replay/util/filesystem.h>
#include <replay/util/image_cache.h>

DEFINE_string(bundler_file, "", "");
DEFINE_string(image_list, "", "");
DEFINE_string(images_directory, "", "");
DEFINE_string(output_reconstruction, "", "");
DEFINE_string(exposure_mesh, "", "");
DEFINE_bool(exposure_align, false, "");

static const int cache_size = 300;

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  replay::Reconstruction scene;
  CHECK(replay::FileExists(FLAGS_bundler_file))
      << "Reconstruction file (" << FLAGS_bundler_file << ") doesn't exist!";
  CHECK(replay::FileExists(FLAGS_image_list))
      << "Reconstruction file (" << FLAGS_image_list << ") doesn't exist!";

  replay::ImageCache images(FLAGS_images_directory, cache_size);
  replay::ReadBundler(FLAGS_bundler_file, FLAGS_image_list, images, &scene);
  CHECK_GT(scene.NumCameras(), 0);

  if (FLAGS_exposure_align) {
    CHECK(replay::FileExists(FLAGS_exposure_mesh))
        << "Mesh needed for exposure alignment.";
    replay::Mesh mesh;
    CHECK(mesh.Load(FLAGS_exposure_mesh));
    CHECK(mesh.NumTriangleFaces() > 0) << "Mesh is empty";
    std::shared_ptr<replay::OpenGLContext> context =
        std::make_shared<replay::OpenGLContext>();
    CHECK(context->Initialize());
    replay::ExposureAlignment::Options exposure_options;
    replay::ExposureAlignment exposure(context, exposure_options, images,
                                       &scene);
    LOG(INFO) << "Aligning exposure";
    int mesh_id = context->UploadMesh(mesh);
    context->BindMesh(mesh_id);
    exposure.GenerateExposureCoefficients();
  }
  LOG(INFO) << "Saving reconstruction.";
  scene.Save(FLAGS_output_reconstruction);
  return 0;
}
