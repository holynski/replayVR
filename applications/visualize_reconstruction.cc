#include <gflags/gflags.h>
#include <glog/logging.h>
#include <replay/camera/pinhole_camera.h>
#include <replay/io/read_bundler.h>
#include <replay/geometry/mesh.h>
#include <replay/multiview/exposure_alignment.h>
#include <replay/rendering/model_renderer.h>
#include <replay/rendering/opengl_context.h>
#include <replay/sfm/reconstruction.h>
#include <replay/util/filesystem.h>
#include <replay/util/image_cache.h>

DEFINE_string(csv_file, "", "");
DEFINE_string(reconstruction_file, "", "");

DEFINE_string(output_folder, "", "");
static const std::string shader_src_dir = REPLAY_SRC_DIR;

cv::Point OrthoProject(const Eigen::Vector3f& center, const float left,
                       const float right, const float top, const float bottom, const int width, const int height) {
  float x = center[0] - left;
  x *= width / (right - left);
  float y = center[2] - top;
  y *= height / (bottom - top);
  return cv::Point(x,y);
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  replay::Reconstruction scene;
  CHECK(scene.Load(FLAGS_reconstruction_file));
  replay::Mesh total_mesh;
  replay::Mesh frustum_mesh = scene.CreateFrustumMesh();
  replay::Mesh point_cloud = scene.CreatePointCloud();
  total_mesh.Append(frustum_mesh);
  total_mesh.Append(point_cloud);
  Eigen::Vector3f min, max;
  total_mesh.GetBoundingBox(&min, &max);
  total_mesh.Save("./test.ply");
  LOG(ERROR) << "Bounding box:";
  LOG(ERROR) << min;
  LOG(ERROR) << max;

  const float l = min[0] - 2;
  const float r = max[0] + 2;
  const float t = min[2] - 2;
  const float b = max[2] + 2;

  int w= 800;
  int h= 800.0 * (r-l) / (b-t);
  cv::Mat3b output(w, h);
  output.setTo(cv::Vec3b(255, 255, 255));

  for (int cam = 0; cam < scene.NumCameras(); cam++) {
    const replay::Camera& camera = scene.GetCamera(cam);
    Eigen::Vector3f position = camera.GetPosition().cast<float>();

    Eigen::Vector3f up = camera.GetUpVector().cast<float>();
    Eigen::Vector3f left = camera.GetRightVector().cast<float>();
    Eigen::Vector3f fwd = camera.GetLookAt().cast<float>();

    Eigen::Vector3f top_left = position + (up / 2) + (left / 2) + fwd;
    Eigen::Vector3f bottom_left = position - (up / 2) + (left / 2) + fwd;
    Eigen::Vector3f top_right = position + (up / 2) - (left / 2) + fwd;
    Eigen::Vector3f bottom_right = position - (up / 2) - (left / 2) + fwd;

    cv::line(output, OrthoProject(position, l, r, t, b, w, h),
             OrthoProject(top_left, l, r, t, b, w, h), cv::Scalar(255, 0, 0));
    cv::line(output, OrthoProject(position, l, r, t, b, w, h),
             OrthoProject(top_right, l, r, t, b, w, h), cv::Scalar(255, 0, 0));
    cv::line(output, OrthoProject(position, l, r, t, b, w, h),
             OrthoProject(bottom_left, l, r, t, b, w, h), cv::Scalar(255, 0, 0));
    cv::line(output, OrthoProject(position, l, r, t, b, w, h),
             OrthoProject(bottom_right, l, r, t, b, w, h), cv::Scalar(255, 0, 0));
    cv::line(output, OrthoProject(bottom_left, l, r, t, b, w, h),
             OrthoProject(bottom_right, l, r, t, b, w, h), cv::Scalar(255, 0, 0));
    cv::line(output, OrthoProject(bottom_left, l, r, t, b, w, h),
             OrthoProject(top_left, l, r, t, b, w, h), cv::Scalar(255, 0, 0));
    cv::line(output, OrthoProject(bottom_right, l, r, t, b, w, h),
             OrthoProject(top_right, l, r, t, b, w, h), cv::Scalar(255, 0, 0));
    cv::line(output, OrthoProject(top_left, l, r, t, b, w, h),
             OrthoProject(top_right, l, r, t, b, w, h), cv::Scalar(255, 0, 0));
  }

  /*
    Eigen::Matrix4f extrinsics = Eigen::Matrix4f::Zero();
    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
    rotation.row(2) = Eigen::Vector3f(0, -1, 0);
    rotation.row(1) = Eigen::Vector3f(1, 0, 0);
    rotation.row(0) = Eigen::Vector3f(0, 0, 1);
    rotation.row(0) = -rotation.row(0);
    rotation.row(2) = -rotation.row(2);
    Eigen::Vector3f position = (min + max) / 2;
    position[1] = max[1] + 1;
    extrinsics.block<3, 3>(0, 0) = rotation;
    extrinsics.block<3, 1>(0, 3) = -rotation * position;
    extrinsics(3, 3) = 1;
    LOG(ERROR) <<  position;

    Eigen::Matrix4f projection;
    const float near = position[1] - max[1];
    const float far = position[1] - min[1];
    projection(0, 0) = 2.0 / (right - left);
    projection(1, 1) = 2.0 / (top - bottom);
    projection(2, 2) = -2.0 / (far - near);
    projection(0, 3) = -(right + left) / (right - left);
    projection(1, 3) = -(top + bottom) / (top - bottom);
    projection(2, 3) = -(far + near) / (far - near);
    projection(3, 3) = 1;

    replay::PinholeCamera camera;
    camera.SetPosition(position.cast<double>());
    camera.SetOrientationFromLookAtUpVector(Eigen::Vector3d(0, -1, 0),
                                            Eigen::Vector3d(1, 0, 0));
    camera.SetFocalLengthFromFOV(Eigen::Vector2d(120, 120));
    Eigen::Vector2i image_size(800, 800);
    camera.SetImageSize(image_size);

    // Implement this as an orthographic projection in OpenGL.
    std::shared_ptr<replay::OpenGLContext> context =
        std::make_shared<replay::OpenGLContext>();
    CHECK(context->Initialize());

    replay::ModelRenderer renderer(context);
    renderer.Initialize();
    int mesh_id = context->UploadMesh(frustum_mesh);
    context->BindMesh(mesh_id);
    context->SetClearColor(Eigen::Vector3f(1, 1, 1));
    // renderer.RenderView(projection, image_size, &output);
    renderer.RenderView(camera, &output);
  */
  cv::imwrite(replay::JoinPath(FLAGS_output_folder, "top_down.png"), output);
  return 0;
}
