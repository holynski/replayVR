#include "replay/rendering/stereo_video_angular_renderer.h"
#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/third_party/theia/sfm/types.h"
#include "replay/util/types.h"
#include "replay/vr_180/vr_180_video_reader.h"
#include "openvr.h"
#include <Eigen/Dense>

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
      "    }"
      "    if (right == 1) {"
      "         frag_uv.x *= 0.5;"
      "         frag_uv.x += 0.5;"
      "    }"
    "}\n";
static const std::string fragment_source =
    "#version 410\n"
    "out vec3 color;\n"
    "in vec3 pos;"
    "in vec2 frag_uv;"
    "uniform sampler2DArray images;"
	"uniform sampler2D left;"
	"uniform sampler2D right;"
    "uniform int image_index;"
    "void main()\n"
    "{\n"
	"    color = texture(left, vec2(0,0)).rgb + texture(right, vec2(0,0)).rgb;"
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
		Eigen::Vector3f& lookat, Eigen::Vector3f& up, Eigen::Matrix3f& rotation) {
  Eigen::AngleAxisf aa(angle_axis.norm(), angle_axis.normalized());
  rotation = aa.toRotationMatrix();
  //rotation.row(2) *= -1;
  //rotation.row(2) *= -1;
  //rotation.row(2) *= -1;
  lookat = rotation.col(2);
  up = rotation.col(1);

  // 0 1 2
  // 0 1
  // 0 2
  // 1 2
  // 1 
  // 0 
  // 2
}

}  // namespace

bool StereoVideoAngularRenderer::Initialize(
    const std::string& spherical_video_filename) {

  if (!renderer_->CompileAndLinkShaders(vertex_source, fragment_source,
                                        &shader_id_)) {
    LOG(ERROR) << "Couldn't compile shader!";
    return false;
  }

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

  static const int number_of_frames = 100;

  CHECK(renderer_->UseShader(shader_id_));
  renderer_->AllocateTextureArray("images", reader_.GetWidth(), reader_.GetHeight(),
                                  image.channels(), number_of_frames);
  frame_lookats_ = std::vector<Eigen::Vector3f>(number_of_frames, Eigen::Vector3f(0,0,0));
  frame_upvecs_ = std::vector<Eigen::Vector3f>(number_of_frames);
  frame_rotations_ = std::vector<Eigen::Matrix3f>(number_of_frames);
  LOG(INFO) << "Uploading frames...";

  int index = 0;
  while (reader_.GetOrientedFrame(image, angle)) {
	  if (index == number_of_frames) {
		  break;
	  }
	  
    CHECK_LT(index, number_of_frames);
    renderer_->UploadTextureToArray(image, "images", index);
    AngleAxisToLookAtUpvec(angle, frame_lookats_[index], frame_upvecs_[index], frame_rotations_[index]);
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

  // set the position as the origin
  camera.SetPosition(Eigen::Vector3d(0, 0, 0));
  // set the viewpoint as negative Z
  Eigen::Matrix3d rotation;
  rotation << 1,0,0,0,-1,0,0,0,-1;
  camera.SetOrientationFromRotationMatrix(rotation);

  int best_frame = -1;
  double best_score = -1;
  for (int i = 0; i < frame_lookats_.size(); i++) {
    const double score = lookat.dot(frame_lookats_[i]);
    if (score > best_score) {
      best_score = score;
      best_frame = i;
    }
  }
  LOG(INFO) << "Rendering from frame " << best_frame;

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
  //renderer_->ShowWindow();
  //renderer_->RenderToWindow();
  renderer_->RenderToImage(&image_);
  cv::imshow("test" + std::to_string(eye_id), image_);
  //cv::imwrite("/Users/holynski/testimageleft.png", image);
  
  renderer_->UploadTexture(image_, eye_id == 0 ? "left" : "right");
  vr::Texture_t eye_texture = { (void*)(uintptr_t)renderer_->GetTextureId(eye_id == 0 ? "left" : "right"), vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
  vr::VRCompositor()->Submit(eye_id == 0 ? vr::Eye_Left : vr::Eye_Right, &eye_texture);
  return false;
}

bool StereoVideoAngularRenderer::RenderEye(Eigen::Matrix4f projection, const int width, const int height,
	const int eye_id,
	const Eigen::Matrix3f& rotation) {
	CHECK(is_initialized_) << "Initialize renderer first.";
	CHECK_LE(eye_id, 1);
	CHECK_GE(eye_id, 0);
	CHECK(renderer_->UseShader(shader_id_));

	Eigen::Matrix3f new_rotation = rotation;
	new_rotation = rotation.inverse();
	//float yaw = atan2(rotation.coeff(2, 1), rotation.coeff(2, 2)); //rotation about X
	//float pitch = atan2(-rotation.coeff(2, 0), sqrt(pow(rotation.coeff(2, 1), 2) + pow(rotation.coeff(2, 2), 2))); //rotation about Y
	//float roll = atan2(rotation.coeff(1, 0), rotation.coeff(0, 0)); //rotation about Z

	//LOG(INFO) << yaw << " " << pitch << " " << roll;
	////yaw *= -1;

	//Eigen::Matrix3f x_rotation;
	//Eigen::Matrix3f y_rotation;
	//Eigen::Matrix3f z_rotation;
	//x_rotation << 1, 0, 0, 0, cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw);
	//y_rotation << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch);
	//z_rotation << cos(roll), -sin(roll), 0, sin(roll), cos(roll), 0, 0, 0, 1;
	//new_rotation = z_rotation * y_rotation * x_rotation;
	//new_rotation = new_rotation.inverse();

	// 12
	// 8
	// 6
	// 1
	// 2

	Eigen::Matrix3f coordinate_change;
	coordinate_change.setIdentity();
	coordinate_change.row(1) *= -1;
		coordinate_change.row(2) *= -1;

	int best_frame = -1;
	double best_score = -1;
	const Eigen::Vector3f lookat = new_rotation.col(2);
	for (int i = 0; i < frame_lookats_.size(); i++) {
		const double score = lookat.dot(frame_lookats_[i]);
		if (score > best_score) {
			best_score = score;
			best_frame = i;
		}
	}
	//LOG(INFO) << "Rendering from frame " << best_frame;
	
	Eigen::Matrix4f mvp;
	mvp.setIdentity();
	Eigen::Matrix3f inverse_rotation = frame_rotations_[best_frame].inverse();
	mvp.block(0, 0, 3, 3) *= new_rotation *  inverse_rotation;

	mvp = projection * mvp;

	CHECK_GE(best_frame, 0);

	renderer_->UploadShaderUniform(best_frame, "image_index");
	renderer_->UploadShaderUniform(eye_id, "right");

	renderer_->UploadMesh(meshes_[eye_id]);

	renderer_->SetProjectionMatrix(mvp);
	renderer_->SetViewportSize(width, height);

	//renderer_->ShowWindow();
	//renderer_->RenderToWindow();

	if (image_.empty()) {
		image_ = cv::Mat3b(1, 1);
	}

	//renderer_->RenderToWindow();
	renderer_->RenderToImage(&image_);
	cv::imshow("test" + std::to_string(eye_id), image_);

	renderer_->UploadTexture(image_, eye_id == 0 ? "left" : "right");
	vr::Texture_t eye_texture = { (void*)(uintptr_t)renderer_->GetTextureId(eye_id == 0 ? "left" : "right"), vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
	vr::VRCompositor()->Submit(eye_id == 0 ? vr::Eye_Left : vr::Eye_Right, &eye_texture);
	return false;
}

}  // namespace replay
