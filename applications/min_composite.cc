#include <gflags/gflags.h>
#include <glog/logging.h>
#include <replay/camera/camera.h>
#include <replay/camera/pinhole_camera.h>
#include <replay/io/read_capreal.h>
#include <replay/rendering/image_based_renderer.h>
#include <replay/rendering/model_renderer.h>
#include <replay/rendering/opengl_context.h>
#include <replay/sfm/reconstruction.h>
#include <replay/util/filesystem.h>
#include <replay/util/image_cache.h>
#include <replay/util/strings.h>

#include <Eigen/Geometry>

DEFINE_string(reconstruction, "/Users/holynski/Drive/research/datasets/windows/london6/internal.txt", "");
DEFINE_string(images_directory, "/Users/holynski/Drive/research/datasets/windows/london6/frames/", "");
DEFINE_string(output_directory, "output/", ""); 
DEFINE_string(mesh, "/Users/holynski/Drive/research/datasets/windows/london6/mesh.ply", ""); 
DEFINE_string(view1, "", "First view to render from.");
DEFINE_string(view2, "", "Second view to render from.");
DEFINE_int32(num_frames, 100, "Number of frames to render.");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  replay::Reconstruction scene;
  CHECK(replay::FileExists(FLAGS_reconstruction))
      << "Reconstruction file (" << FLAGS_reconstruction << ") doesn't exist!";

  replay::ImageCache images(FLAGS_images_directory, 20);
  replay::ReadCapreal(FLAGS_reconstruction, images, &scene);

  CHECK_GT(scene.NumCameras(), 0);

  LOG(INFO) << "Loading the mesh.";
  CHECK(replay::FileExists(FLAGS_mesh));
  replay::Mesh mesh;
  CHECK(mesh.Load(FLAGS_mesh));
  CHECK(mesh.NumTriangleFaces() > 0) << "Mesh is empty";

  std::shared_ptr<replay::OpenGLContext> context =
      std::make_shared<replay::OpenGLContext>();
  CHECK(context->Initialize());

  replay::ImageBasedRenderer projection_renderer(context);
  //int mesh_id = context->UploadMesh(mesh);
  CHECK(projection_renderer.Initialize(scene, images, mesh));

  const replay::Camera& viewpoint1 = scene.GetCamera(0);
  const replay::Camera& viewpoint2 = scene.GetCamera(scene.NumCameras() - 1);

  replay::Camera* viewpoint = viewpoint1.Clone();

  int k = 0;
  int start_index = 0;
  int end_index = scene.NumCameras();
  for (int i = start_index; i < end_index; i++) {
    // Set the interpolated position.
    const double interpolation_weight =
        static_cast<double>(i) / static_cast<double>(FLAGS_num_frames);
    viewpoint->SetPosition((1.0 - interpolation_weight) *
                               viewpoint1.GetPosition() +
                           interpolation_weight * viewpoint2.GetPosition());

    viewpoint->SetOrientation(viewpoint1.GetOrientation().slerp(
        interpolation_weight, viewpoint2.GetOrientation()));

    cv::Mat3b minc;
    CHECK(projection_renderer.RenderView(scene.GetCamera(i), &minc, i));
    cv::imwrite(FLAGS_output_directory + "/minc_" + replay::PadZeros(k,6) + ".png",
                minc);
    k++;
  }


}
