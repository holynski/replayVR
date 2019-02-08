#include <gflags/gflags.h>
#include <glog/logging.h>
#include <replay/camera/camera.h>
#include <replay/camera/pinhole_camera.h>
#include <replay/io/read_capreal.h>
#include <replay/rendering/image_based_renderer.h>
#include <replay/rendering/opengl_context.h>
#include <replay/sfm/reconstruction.h>
#include <replay/util/filesystem.h>
#include <replay/util/image_cache.h>

#include <Eigen/Geometry>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  std::shared_ptr<replay::OpenGLContext> context =
      std::make_shared<replay::OpenGLContext>();
  CHECK(context->Initialize());

  int shader_id = 0;
  const std::string fragment_source =
      "#version 410\n \n \
       out vec3 color;\
       uniform int time;\
       void main() {\
         color = vec3((time % 5000) / 5000.0, (time % 1100) / 1100.0, (time % 4000) / 4000.0);\
}";
  CHECK(context->CompileFullScreenShader(fragment_source, &shader_id));
  context->SetViewportSize(500, 500);
  context->UseShader(shader_id);
  context->ShowWindow();
  int time = 0;
  while (true) {
    context->UploadShaderUniform(time++, "time");
    context->Render();
  }
}
