#include "replay/rendering/stereo_video_angular_renderer.h"
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/third_party/theia/sfm/types.h"
#include "replay/util/types.h"
#include "replay/vr_180/vr_180_video_reader.h"

namespace replay {
  namespace {
    static const std::string vertex_source =
      "#version 410\n"
      "uniform mat4 MVP;\n"
      "in vec3 vert;"
      "in vec2 uv;"
      "out vec3 pos;"
      "out vec2 frag_uv;"
      "uniform int right;"
      "void main()\n"
      "{\n"
      "    gl_Position = MVP * vec4(vert, 1.0);\n"
      "    pos = vert;"
      "    frag_uv = uv;"
      "    if (right == 0) {"
      "         frag_uv.x *= 0.5;"
      "         frag_uv.x = 1.0 - frag_uv.x;"
      "         frag_uv.x -= 0.5;"
      "    }"
      "    if (right > 0) {"
      "         frag_uv.x *= 0.5;"
      "         frag_uv.x += 0.5;"
      "         frag_uv.y = 1.0 - frag_uv.y;"
      "    }"
    "}\n";
static const std::string fragment_source =
    "#version 410\n"
    "out vec3 color;\n"
    "in vec3 pos;"
    "in vec2 frag_uv;"
    "uniform sampler2DArray images;"
    "uniform int image_index;"
    "void main()\n"
    "{\n"
    "    color = texture(images, vec3(frag_uv, image_index)).rgb;"
    "}\n";
}  // namespace

StereoVideoAngularRenderer::StereoVideoAngularRenderer(
    std::shared_ptr<OpenGLRenderer> renderer)
    : renderer_(renderer), is_initialized_(false) {
  CHECK(renderer->IsInitialized()) << "Initialize OpenGL renderer first!";
}

namespace {

void AngleAxisToLookAtUpvec(const Eigen::Vector3f& angle_axis,
                            Eigen::Vector3f& lookat, Eigen::Vector3f& up) {
  Eigen::AngleAxisf aa(angle_axis.norm(), angle_axis.normalized());
  lookat = aa.toRotationMatrix().col(2);
  up = aa.toRotationMatrix().col(1);
}

}  // namespace

bool StereoVideoAngularRenderer::Initialize(
    const std::string& spherical_video_filename) {
  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(ERROR) << "Couldn't compile shader!";
    return false;
  }
  //if (!renderer_->CompileFullScreenShader(fragment_source,
                                        //&shader_id_)) {
    //LOG(ERROR) << "Couldn't compile shader!";
    //return false;
  //}
  renderer_->SetViewportSize(1000,1000);

  renderer_->HideWindow();
  is_initialized_ = true;
  if (!reader_.Open(spherical_video_filename)) {
    LOG(ERROR) << "Couldn't open spherical video file: "
               << spherical_video_filename;
    return false;
  }

  meshes_ = reader_.GetMeshes();

  if (meshes_.size() != 2) {
    LOG(ERROR) << "No left/right projection meshes.";
    return false;
  }

  cv::Mat3b image;
  Eigen::Vector3f angle;

  CHECK(renderer_->UseShader(shader_id_));
  renderer_->AllocateTextureArray("images", reader_.GetWidth(), reader_.GetHeight(),
                                  image.channels(), 285);
  frame_lookats_ = std::vector<Eigen::Vector3f>(285);
  frame_upvecs_ = std::vector<Eigen::Vector3f>(285);

  LOG(INFO) << "Uploading frames...";

  int index = 0;
  while (reader_.GetOrientedFrame(image, angle)) {
    CHECK_LT(index, 285);
    renderer_->UploadTextureToArray(image, "images", index);
    AngleAxisToLookAtUpvec(angle, frame_lookats_[index], frame_upvecs_[index]);
        index++;
  }

  LOG(INFO) << "Done";

  return true;
}

bool StereoVideoAngularRenderer::RenderEye(theia::Camera camera,
                                           const int eye_id,
                                           const Eigen::Vector3f& lookat) {
  CHECK(is_initialized_) << "Initialize renderer first.";
  CHECK_LE(eye_id, 1);
  CHECK_GE(eye_id, 0);
  CHECK(renderer_->UseShader(shader_id_));
  LOG(INFO) << "Using shader " << shader_id_;

  // set the position as the origin
  camera.SetPosition(Eigen::Vector3d(0, 0, 0));
  // set the viewpoint as negative Z
  Eigen::Matrix3d rotation;
  rotation << -1,0,0,0,1,0,0,0,-1;
  camera.SetOrientationFromRotationMatrix(rotation);

  int best_frame = -1;
  double best_score = 0;
  for (int i = 0; i < frame_lookats_.size(); i++) {
    const double score = lookat.dot(frame_lookats_[i]);
    if (score > best_score) {
      best_score = score;
      best_frame = i;
    }
  }

  CHECK_GE(best_frame, 0);

  renderer_->UploadShaderUniform(best_frame, "image_index");
  renderer_->UploadShaderUniform(eye_id, "right");

  renderer_->UploadMesh(meshes_[eye_id]);

  renderer_->SetViewpoint(camera);

  //renderer_->ShowWindow();
  //renderer_->RenderToWindow();

  if (image_.empty()) {
    image_ = cv::Mat3b(1,1);
  }
  renderer_->ShowWindow();
  renderer_->RenderToWindow();
  //renderer_->RenderToImage(&image_);

  //cv::imshow("test" + std::to_string(eye_id), image_);
  //cv::imwrite("/Users/holynski/testimageleft.png", image);
  return false;
}

}  // namespace replay
