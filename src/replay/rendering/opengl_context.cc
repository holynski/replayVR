#include "replay/rendering/opengl_context.h"

#ifdef __APPLE__
#define GLFW_INCLUDE_GLCOREARB
#else  // __APPLE__
#include <GL/glew.h>
#endif  // __APPLE__
#include <GLFW/glfw3.h>

#include <glog/logging.h>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "replay/depth_map/depth_map.h"
#include "replay/geometry/mesh.h"
#include "replay/geometry/triangle_id_map.h"
#include "replay/util/row_array.h"

namespace replay {
namespace {

std::string OpenGLErrorCodeToString(GLenum error) {
  switch (error) {
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      return "GL_INVALID_FRAMEBUFFER_OPERATION";
    default:
      return "UNKNOWN_ERROR (" + std::to_string(error) + ")";
  };
}

// TODO(holynski): The following does not work:
//  mesh_id = UploadMesh(mesh);
//  UseShader(A);
//  BindMesh(mesh_id);
//  Render();
//  UseShader(B);
//  BindMesh(mesh_id);
//  Render();

// void PrintAvailableMemoryNvidia() {
// static const int GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX = 0x9048;
// static const int GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX  = 0x9049;
// GLint nTotalMemoryInKB = 0;
// glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX,
//&nTotalMemoryInKB);
// GLint nCurAvailMemoryInKB = 0;
// glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX,
//&nCurAvailMemoryInKB);

// LOG(INFO) << "Available GPU memory: " << nCurAvailMemoryInKB / 1000 << "MB /
// " << nTotalMemoryInKB / 1000 << "MB";
//}

void CheckForOpenGLErrors() {
  int error = glGetError();
  if (error != 0) {
    LOG(FATAL) << "OpenGL returned error: " << OpenGLErrorCodeToString(error);
  }
}

void GLFWErrorCallback(int error, const char* description) {
  LOG(ERROR) << "Error " << error << ": " << description;
}

}  // namespace

int OpenGLContext::instantiated_renderers_ = 0;
float OpenGLContext::framebuffer_size_to_screen_coords_ = 1.0;
std::unordered_map<GLFWwindow*, OpenGLContext*>
    OpenGLContext::window_to_renderer_;

OpenGLContext::OpenGLContext()
    : fullscreen_triangle_mesh_(-1),
      pbo_id_(-1),
      mvp_location_(-1),
      is_initialized_(false),
      window_showing_(false),
      clear_color_(0, 0, 0) {}

OpenGLContext::~OpenGLContext() {
  instantiated_renderers_--;
  if (instantiated_renderers_ <= 0) {
    DestroyContext();
  }
}

void OpenGLContext::DestroyContext() {
  glfwMakeContextCurrent(window_);
  for (int i = 0; i < programs_.size(); i++) {
    glDeleteProgram(programs_[i]);
    glDeleteShader(fragment_shaders_[i]);
    glDeleteShader(vertex_shaders_[i]);
  }
  for (int i = 0; i < ebos_.size(); i++) {
    glDeleteBuffers(1, &ebos_[current_mesh_]);
    glDeleteBuffers(1, &vbos_[current_mesh_]);
    glDeleteVertexArrays(1, &vaos_[current_mesh_]);
  }
  glfwTerminate();
}

bool OpenGLContext::ReadShaderFromFile(const std::string& filename,
                                       std::string* shader_src) {
  std::ifstream shader_ifstream(filename.c_str());
  if (!shader_ifstream.is_open()) {
    return false;
  }

  // Get shader source
  shader_src->assign(std::istreambuf_iterator<char>(shader_ifstream),
                     std::istreambuf_iterator<char>());
  return true;
}

bool OpenGLContext::CompileAndLinkShaders(const std::string& vertex,
                                          const std::string& fragment,
                                          int* shader_id) {
  glfwMakeContextCurrent(window_);
  CHECK(is_initialized_) << "Call initialize first.";
  GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  const GLchar* source = (const GLchar*)vertex.c_str();
  glShaderSource(vertex_shader, 1, &source, 0);
  glCompileShader(vertex_shader);
  GLint is_compiled = GL_FALSE;
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &is_compiled);
  if (is_compiled == GL_FALSE) {
    GLint max_length = 0;
    glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &max_length);
    std::vector<GLchar> info_log(max_length);
    glGetShaderInfoLog(vertex_shader, max_length, &max_length, &info_log[0]);
    glDeleteShader(vertex_shader);
    LOG(ERROR) << "Compiling vertex shader failed: " << info_log.data();
    return false;
  }
  GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  source = (const GLchar*)fragment.c_str();
  glShaderSource(fragment_shader, 1, &source, 0);

  glCompileShader(fragment_shader);
  glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &is_compiled);
  if (is_compiled == GL_FALSE) {
    GLint max_length = 0;
    glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &max_length);

    std::vector<GLchar> info_log(max_length);
    glGetShaderInfoLog(fragment_shader, max_length, &max_length, &info_log[0]);
    glDeleteShader(fragment_shader);
    glDeleteShader(vertex_shader);
    LOG(ERROR) << "Compiling fragment shader failed: " << info_log.data();
    return false;
  }

  GLuint program = glCreateProgram();

  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);

  GLint is_linked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, (int*)&is_linked);
  if (is_linked == GL_FALSE) {
    GLint max_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);
    std::vector<GLchar> info_log(max_length);
    glGetProgramInfoLog(program, max_length, &max_length, &info_log[0]);
    glDeleteProgram(program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    LOG(ERROR) << "Linking shaders failed: " << info_log.data();
    return false;
  }
  glDetachShader(program, vertex_shader);
  glDetachShader(program, fragment_shader);
  // glBindAttribLocation(program, 0, "vert");
  // if (vertex.find("in vec2 vertexUV;") != std::string::npos) {
  // glBindAttribLocation(program, 2, "vertexUV");
  //}
  glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
  programs_.push_back(program);
  fragment_shaders_.push_back(fragment_shader);
  vertex_shaders_.push_back(vertex_shader);
  buffers_bound_.push_back(false);
  output_buffers_.push_back(0);
  if (output_buffers_.size() > using_projection_matrix_.size()) {
    using_projection_matrix_.push_back(true);
  }
  *shader_id = programs_.size() - 1;
  CheckForOpenGLErrors();
  instantiated_renderers_++;
  return true;
}

bool OpenGLContext::CompileFullScreenShader(const std::string& fragment,
                                            int* shader_id) {
  // Create a full-screen mesh. Upload this mesh so that it clears all vertex
  // and element buffers.

  // Load the full screen vertex shader to a string.
  static const std::string full_screen_vs =
      "#version 410\n"
      "uniform mat4 MVP;\n"
      "uniform float negative;\n"
      "in vec3 vert;\n"
      "void main()\n"
      "{\n"
      "  gl_Position = vec4(vert.x, vert.y * negative, vert.z, 1.0);\n"
      "}\n";
  using_projection_matrix_.push_back(false);

  // Compile and link the shaders per usual.
  if (!CompileAndLinkShaders(full_screen_vs, fragment, shader_id)) {
    return false;
  }

  return true;
}

bool OpenGLContext::IsInitialized() const { return is_initialized_; }

bool OpenGLContext::Initialize() {
  Keyboard_ = NULL;
  MouseButton_ = NULL;
  MouseMove_ = NULL;
  current_program_ = -1;
  current_mesh_ = -1;
  if (is_initialized_) {
    return true;
  }
  projection_.setIdentity();

  glfwSetErrorCallback(GLFWErrorCallback);
  if (!glfwInit()) {
    LOG(ERROR) << "Could not initialize GLFW";
    return false;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window_ = glfwCreateWindow(1920, 1080, "OpenGL Window", NULL, NULL);
  if (window_ == nullptr) {
    LOG(FATAL) << "Window could not be created.";
    return false;
  }
  glfwMakeContextCurrent(window_);
  int screen_width, screen_height;
  glfwSetWindowSize(window_, 1920, 1080);
  glfwGetFramebufferSize(window_, &screen_width, &screen_height);
  framebuffer_size_to_screen_coords_ = (float)1920.0f / (float)screen_width;
  window_showing_ = false;
  glfwSwapBuffers(window_);
  glfwPollEvents();
#ifndef __APPLE__
  const GLenum err = glewInit();
  if (err != GLEW_OK) {
    LOG(ERROR) << "glewInit() failed!\nError: " << glewGetErrorString(err);
    return false;
  }
#endif  // __APPLE__
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glFrontFace(GL_CCW);
  is_initialized_ = true;

  CheckForOpenGLErrors();
  window_to_renderer_[window_] = this;
  glfwSetKeyCallback(window_, &OpenGLContext::KeyboardCallback);
  glfwSetMouseButtonCallback(window_, &OpenGLContext::MouseButtonCallback);
  glfwSetCursorPosCallback(window_, &OpenGLContext::MousePosCallback);
  LOG(INFO) << "OpenGL Version: " << glGetString(GL_VERSION);
  return true;
}

void OpenGLContext::MouseButtonCallback(GLFWwindow* window, int button,
                                        int action, int mods) {
  if (window == NULL) {
    return;
  }
  OpenGLContext* renderer = window_to_renderer_[window];
  if (renderer == NULL) {
    return;
  }
  if (renderer->MouseButton_ == NULL) {
    return;
  }
  (*(renderer->MouseButton_))(button, action, mods);
}

void OpenGLContext::KeyboardCallback(GLFWwindow* window, int key, int scancode,
                                     int action, int mods) {
  if (window == NULL) {
    return;
  }
  OpenGLContext* renderer = window_to_renderer_[window];
  if (renderer == NULL) {
    return;
  }
  if (renderer->Keyboard_ == NULL) {
    return;
  }
  renderer->Keyboard_(key, action, mods);
}

void OpenGLContext::MousePosCallback(GLFWwindow* window, double x, double y) {
  if (window == NULL) {
    return;
  }
  OpenGLContext* renderer = window_to_renderer_[window];
  if (renderer == NULL) {
    return;
  }
  if (renderer->MouseMove_ == NULL) {
    return;
  }
  renderer->mouse_position_[0] = x;
  renderer->mouse_position_[1] = y;
  (*(renderer->MouseMove_))(x / framebuffer_size_to_screen_coords_,
                            y / framebuffer_size_to_screen_coords_);
}

bool OpenGLContext::SetMousePositionCallback(void (*callback)(double, double)) {
  MouseMove_ = callback;
  return true;
}

bool OpenGLContext::SetMouseClickCallback(void (*callback)(int, int, int)) {
  MouseButton_ = callback;
  LOG(INFO) << "Setting callback";
  return true;
}

bool OpenGLContext::SetKeyboardCallback(
    std::function<void(int, int, int)> callback) {
  Keyboard_ = callback;
  return true;
}

Eigen::Vector2d OpenGLContext::GetMousePosition() const {
  return mouse_position_;
}

bool OpenGLContext::BindFullscreenTriangle() {
  if (fullscreen_triangle_mesh_ < 0) {
    Mesh single_triangle_mesh;
    single_triangle_mesh.AddVertex(Eigen::Vector3f(-1, -1, 0));
    single_triangle_mesh.AddVertex(Eigen::Vector3f(3, -1, 0));
    single_triangle_mesh.AddVertex(Eigen::Vector3f(-1, 3, 0));
    single_triangle_mesh.AddTriangleFace(0, 1, 2);
    fullscreen_triangle_mesh_ = UploadMesh(single_triangle_mesh);
    if (fullscreen_triangle_mesh_ < 0) {
      return false;
    }
  }
  return true;
  // return BindMesh(fullscreen_triangle_mesh_);
}

int OpenGLContext::UploadMesh(const Mesh& mesh) {
  glfwMakeContextCurrent(window_);
  if (mesh.NumVertices() <= 0) {
    return -1;
  }

  const GLint vert_location =
      glGetAttribLocation(programs_[current_program_], "vert");
  GLuint vao, vbo, ebo, uvbo, cbo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glEnableVertexAttribArray(vert_location);

  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  glGenBuffers(1, &uvbo);
  glGenBuffers(1, &cbo);

  vaos_.push_back(vao);
  vbos_.push_back(vbo);
  ebos_.push_back(ebo);
  uvbos_.push_back(uvbo);
  cbos_.push_back(cbo);

  int mesh_id = vaos_.size() - 1;

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, mesh.NumVertices() * 3 * 4,
               (void*)mesh.vertex_positions(), GL_STATIC_DRAW);

  glVertexAttribPointer(vert_location, 3, GL_FLOAT, GL_FALSE, 0, 0);

  if (mesh.HasUVs()) {
    const GLint uv_location =
        glGetAttribLocation(programs_[current_program_], "uv");
    glBindBuffer(GL_ARRAY_BUFFER, uvbo);
    const float* uvs = mesh.uvs();
    glBufferData(GL_ARRAY_BUFFER, mesh.NumVertices() * 4 * 2, (void*)uvs,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(uv_location);
    glBindBuffer(GL_ARRAY_BUFFER, uvbo);
    glVertexAttribPointer(uv_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
  }

  if (mesh.HasColors()) {
    const GLint color_location =
        glGetAttribLocation(programs_[current_program_], "rgb");
    if (color_location >= 0) {
      glBindBuffer(GL_ARRAY_BUFFER, cbo);
      const float* rgbs = mesh.colors();
      glBufferData(GL_ARRAY_BUFFER, mesh.NumVertices() * 4 * 3, (void*)rgbs,
                   GL_STATIC_DRAW);
      glEnableVertexAttribArray(color_location);
      glBindBuffer(GL_ARRAY_BUFFER, cbo);
      glVertexAttribPointer(color_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
    }
  }

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.NumTriangleFaces() * 3 * 4,
               (void*)mesh.triangles().data(), GL_STATIC_DRAW);

  num_triangles_.push_back(mesh.triangles().size());
  CheckForOpenGLErrors();
  return mesh_id;
}

bool OpenGLContext::UpdateMesh(const Mesh& mesh, const int mesh_id) {
  LOG(FATAL) << "Not implemented";
  return false;
}

bool OpenGLContext::BindMesh(const int mesh_id) {
  CHECK(is_initialized_) << "Call initialize first.";
  if (mesh_id < 0 || mesh_id >= vaos_.size()) {
    return false;
  }
  current_mesh_ = mesh_id;
  const GLint vert_location =
      glGetAttribLocation(programs_[current_program_], "vert");

  glBindVertexArray(vaos_[current_mesh_]);
  glEnableVertexAttribArray(vert_location);

  glBindVertexArray(vaos_[current_mesh_]);
  glBindBuffer(GL_ARRAY_BUFFER, vbos_[current_mesh_]);

  glVertexAttribPointer(vert_location, 3, GL_FLOAT, GL_FALSE, 0, 0);

  const GLint uv_location =
      glGetAttribLocation(programs_[current_program_], "uv");
  if (uv_location >= 0) {
    glBindBuffer(GL_ARRAY_BUFFER, uvbos_[current_mesh_]);
    glEnableVertexAttribArray(uv_location);
    glBindBuffer(GL_ARRAY_BUFFER, uvbos_[current_mesh_]);
    glVertexAttribPointer(uv_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
  }

  const GLint color_location =
      glGetAttribLocation(programs_[current_program_], "rgb");
  if (color_location >= 0) {
    glBindBuffer(GL_ARRAY_BUFFER, cbos_[current_mesh_]);
    glEnableVertexAttribArray(color_location);
    glBindBuffer(GL_ARRAY_BUFFER, cbos_[current_mesh_]);
    glVertexAttribPointer(color_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
  }

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos_[current_mesh_]);

  return true;
}

void OpenGLContext::UploadShaderUniform(const int& val,
                                        const std::string& name) {
  glfwMakeContextCurrent(window_);
  glUseProgram(programs_[current_program_]);
  GLint location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniform1i(location, val);
  CheckForOpenGLErrors();
}

void OpenGLContext::UploadShaderUniform(const Eigen::Vector3f& val,
                                        const std::string& name) {
  glfwMakeContextCurrent(window_);
  glUseProgram(programs_[current_program_]);
  GLint location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniform3fv(location, 1, val.data());
  CheckForOpenGLErrors();
}

void OpenGLContext::UploadShaderUniform(const Eigen::Vector2f& val,
                                        const std::string& name) {
  glfwMakeContextCurrent(window_);
  glUseProgram(programs_[current_program_]);
  GLint location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniform2fv(location, 1, val.data());
  CheckForOpenGLErrors();
}

void OpenGLContext::UploadShaderUniform(const float& val,
                                        const std::string& name) {
  glfwMakeContextCurrent(window_);
  glUseProgram(programs_[current_program_]);
  GLint location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniform1fv(location, 1, &val);
  CheckForOpenGLErrors();
}

void OpenGLContext::UploadShaderUniform(const Eigen::Matrix4f& val,
                                        const std::string& name) {
  glfwMakeContextCurrent(window_);
  glUseProgram(programs_[current_program_]);
  GLint location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniformMatrix4fv(location, 1, GL_FALSE, val.data());
  CheckForOpenGLErrors();
}

void OpenGLContext::UploadShaderUniform(const std::vector<Eigen::Matrix4f>& val,
                                        const std::string& name) {
  glfwMakeContextCurrent(window_);
  glUseProgram(programs_[current_program_]);
  GLint location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniformMatrix4fv(location, val.size(), GL_FALSE, val[0].data());
  CheckForOpenGLErrors();
}

void OpenGLContext::UploadShaderUniform(const std::vector<Eigen::Vector3f>& val,
                                        const std::string& name) {
  glfwMakeContextCurrent(window_);
  glUseProgram(programs_[current_program_]);
  GLint location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniform3fv(location, val.size(), val[0].data());
  CheckForOpenGLErrors();
}

bool OpenGLContext::CreateRenderBuffer(const int& datatype, const int& format) {
  glfwMakeContextCurrent(window_);
  int internal_format;
  switch (datatype) {
    case GL_FLOAT:
      switch (format) {
        case GL_RGB:
          internal_format = GL_RGB32F;
          break;
        case GL_RED:
          internal_format = GL_R32F;
          break;
        case GL_RGBA:
          internal_format = GL_RGBA32F;
          break;
        case GL_RG:
          internal_format = GL_RG32F;
          break;
        default:
          internal_format = 0;
          LOG(ERROR) << "Buffer type not supported.";
          return false;
      }
      break;
    case GL_UNSIGNED_BYTE:
      switch (format) {
        case GL_RGB:
          internal_format = GL_RGB8;
          break;
        case GL_RED:
          internal_format = GL_R8;
          break;
        case GL_RGBA:
          internal_format = GL_RGBA8;
          break;
        default:
          internal_format = 0;
          LOG(ERROR) << "Buffer type not supported.";
          return false;
      }
      break;
    case GL_UNSIGNED_INT:
      switch (format) {
        case GL_RED_INTEGER:
          internal_format = GL_R32UI;
          break;
        case GL_RGB:
          internal_format = GL_RGB32UI;
          break;
        case GL_RGBA:
          internal_format = GL_RGBA32UI;
        default:
          internal_format = 0;
          LOG(ERROR) << "Buffer type not supported.";
          return false;
      }
      break;
    default:
      internal_format = 0;
      LOG(ERROR) << "Buffer type not supported.";
      return false;
  }

  if (buffer_types_[internal_format] > 0) {
    output_buffers_[current_program_] =
        output_buffers_[buffer_types_[internal_format]];
    buffers_bound_[current_program_] = true;
    return true;
  }

  glActiveTexture(GL_TEXTURE0);
  GLuint output_texture = 0;
  GLuint output_buffer = 0;
  GLuint render_buffer;
  glGenFramebuffers(1, &output_buffer);
  glGenTextures(1, &output_texture);
  glGenRenderbuffers(1, &render_buffer);
  glBindFramebuffer(GL_FRAMEBUFFER, output_buffer);
  glBindTexture(GL_TEXTURE_2D, output_texture);

  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, 5000, 5000, 0, format,
               datatype, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         output_texture, 0);
  glBindRenderbuffer(GL_RENDERBUFFER, render_buffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, 5000, 5000);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, render_buffer);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(ERROR) << "Framebuffer could not be created.";
    return false;
  }

  int pbo_size = 5000 * 5000 * 4 * 4;
  glGenBuffers(1, &pbo_id_);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id_);
  glBufferData(GL_PIXEL_PACK_BUFFER, pbo_size, 0, GL_DYNAMIC_READ);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  CheckForOpenGLErrors();

  output_buffers_[current_program_] = output_buffer;
  buffers_bound_[current_program_] = true;
  buffer_types_[internal_format] = current_program_;
  return true;
}

bool OpenGLContext::UploadTextureInternal(void* data, const int& width,
                                          const int& height, const int& format,
                                          const int& datatype,
                                          const int& internal_format,
                                          const std::string& name) {
  glfwMakeContextCurrent(window_);
  CHECK(current_program_ >= 0) << "Did not call UseShader!";
  glUseProgram(programs_[current_program_]);
  CheckForOpenGLErrors();
  // if (textures_.count(name) > 0) {
  // return UpdateTextureInternal(data, width,
  // height, format,
  // datatype,
  // internal_format,
  // name);
  //}
  if (textures_.count(name) < 1) {
    GLuint tex;
    glGenTextures(1, &tex);
    textures_opengl_[name] = textures_.size();
    textures_[name] = tex;
    CheckForOpenGLErrors();
  }
  GLuint tex = textures_[name];
  // This might cause trouble ^. Why is it being set to the last one always?
  glActiveTexture(GL_TEXTURE4 + textures_opengl_[name]);
  CheckForOpenGLErrors();
  glBindTexture(GL_TEXTURE_2D, tex);
  CheckForOpenGLErrors();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  CheckForOpenGLErrors();
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format,
               datatype, data);
  CheckForOpenGLErrors();
  GLint texture_location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniform1i(texture_location, 4 + textures_opengl_[name]);

  CheckForOpenGLErrors();
  BindMesh(current_mesh_);
  return true;
}

GLuint OpenGLContext::GetTextureId(const std::string& name) const {
  GLuint tex = textures_.at(name);
  return tex;
}

bool OpenGLContext::UploadTexture(const cv::Mat& image,
                                  const std::string& name) {
  cv::Mat dst = image.clone();
  // cv::flip(image, dst, 0);
  if (dst.rows % 2 != 0 || dst.cols % 2 != 0) {
    cv::resize(dst, dst,
               cv::Size(dst.cols - (dst.cols % 2), dst.rows - (dst.rows % 2)));
  }
  unsigned char mat_type = image.type() & CV_MAT_DEPTH_MASK;
  GLint datatype;
  switch (mat_type) {
    case CV_8U:
      datatype = GL_UNSIGNED_BYTE;
      break;
    case CV_32F:
      datatype = GL_FLOAT;
      break;
    default:
      LOG(FATAL) << "Mat type not supported.";
  }
  if (dst.channels() == 3) {
    return UploadTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                 dst.rows, GL_RGB, datatype, GL_RGB8, name);
  }
  if (dst.channels() == 4) {
    return UploadTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                 dst.rows, GL_RGBA, datatype, GL_RGBA8, name);
  }
  if (dst.channels() == 1) {
    return UploadTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                 dst.rows, GL_RED, datatype, GL_R8, name);
  }
  if (dst.channels() == 2) {
    if (datatype == GL_FLOAT) {
      return UploadTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                   dst.rows, GL_RG, datatype, GL_RG32F, name);
    } else {
      return UploadTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                   dst.rows, GL_RG, datatype, GL_RG16, name);
    }
  }
  LOG(FATAL) << "Number of channels " << dst.channels()
             << " not implemented in UploadTexture.";
  return false;
}

bool OpenGLContext::UpdateTextureInternal(void* data, const int& width,
                                          const int& height, const int& format,
                                          const int& datatype,
                                          const int& internal_format,
                                          const std::string& name) {
  glfwMakeContextCurrent(window_);
  DCHECK(current_program_ >= 0) << "Did not call UseShader!";
  glUseProgram(programs_[current_program_]);
  if (textures_.count(name) < 1) {
    return false;
  }
  glActiveTexture(GL_TEXTURE4 + textures_opengl_[name]);
  GLuint tex = textures_[name];
  glBindTexture(GL_TEXTURE_2D, tex);
  int w, h;
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);

  // If the texture size has changed, we need to delete this texture and create
  // a new one.
  if (width != w || height != h) {
    // glDeleteTextures(1, &tex);
    textures_opengl_.erase(name);
    textures_.erase(name);
    return UploadTextureInternal(data, width, height, format, datatype,
                                 internal_format, name);
  }
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, datatype,
                  data);
  CheckForOpenGLErrors();
  return true;
}

bool OpenGLContext::UpdateTexture(const cv::Mat& image,
                                  const std::string& name) {
  cv::Mat dst = image.clone();
  // cv::flip(image, dst, 0);
  if (dst.rows % 2 != 0 || dst.cols % 2 != 0) {
    cv::resize(dst, dst,
               cv::Size(dst.cols - (dst.cols % 2), dst.rows - (dst.rows % 2)));
  }
  if (dst.channels() == 3)
    return UpdateTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                 dst.rows, GL_RGB, GL_UNSIGNED_BYTE, GL_RGB8,
                                 name);
  if (dst.channels() == 4)
    return UpdateTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                 dst.rows, GL_RGBA, GL_UNSIGNED_BYTE, GL_RGBA8,
                                 name);
  if (dst.channels() == 1)
    return UpdateTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                 dst.rows, GL_RED, GL_UNSIGNED_BYTE, GL_R8,
                                 name);
  if (dst.channels() == 2)
    return UpdateTextureInternal(reinterpret_cast<void*>(dst.data), dst.cols,
                                 dst.rows, GL_RG, GL_UNSIGNED_BYTE, GL_RG8,
                                 name);
  LOG(FATAL) << "Number of channels " << dst.channels()
             << " not implemented in UploadTexture.";
  return false;
}

bool OpenGLContext::UploadTexture(const DepthMap& depth,
                                  const std::string& name) {
  return UploadTextureInternal((void*)depth.Depth().data, depth.Cols(),
                               depth.Rows(), GL_RED, GL_FLOAT, GL_R32F, name);
}

bool OpenGLContext::AllocateTextureArray(const std::string& name,
                                         const int& width, const int& height,
                                         const int& channels,
                                         const int& num_elements,
                                         const bool compressed,
                                         const GLint datatype) {
  glfwMakeContextCurrent(window_);
  CHECK(current_program_ >= 0) << "Did not call UseShader!";
  glUseProgram(programs_[current_program_]);

  if (textures_.count(name) < 1) {
    GLuint tex;
    glGenTextures(1, &tex);
    textures_opengl_[name] = textures_.size();
    textures_[name] = tex;
  }

  GLuint tex = textures_[name];
  glActiveTexture(GL_TEXTURE4 + textures_opengl_[name]);
  glBindTexture(GL_TEXTURE_2D_ARRAY, tex);

  switch (channels) {
    case 1:
      glTexImage3D(GL_TEXTURE_2D_ARRAY, 0,
                   (compressed ? GL_COMPRESSED_RED : GL_RED), width, height,
                   num_elements, 0, GL_RED, datatype, NULL);
      break;
    case 3:
      glTexImage3D(GL_TEXTURE_2D_ARRAY, 0,
                   (compressed ? GL_COMPRESSED_RGB : GL_RGB), width, height,
                   num_elements, 0, GL_RGB, datatype, NULL);
      break;
    case 4:
      glTexImage3D(GL_TEXTURE_2D_ARRAY, 0,
                   (compressed ? GL_COMPRESSED_RGBA : GL_RGBA), width, height,
                   num_elements, 0, GL_RGBA, datatype, NULL);
      break;
    default:
      LOG(FATAL) << "Texture array of " << channels
                 << " channels is not implemented.";
      break;
  }

  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  GLint texture_location =
      glGetUniformLocation(programs_[current_program_], name.c_str());
  glUniform1i(texture_location, 4 + textures_opengl_[name]);
  CheckForOpenGLErrors();
  return true;
}

bool OpenGLContext::UploadTextureToArrayInternal(
    const void* data, const std::string& name, const int& width,
    const int& height, const int& format, const GLint& datatype,
    const int& index) {
  glfwMakeContextCurrent(window_);
  CHECK(current_program_ >= 0) << "Did not call UseShader!";
  glUseProgram(programs_[current_program_]);
  CHECK(textures_.count(name) >= 1) << "Did not call AllocateTextureArray!";
  glActiveTexture(GL_TEXTURE4 + textures_opengl_[name]);
  GLuint tex = textures_[name];
  glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
  glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, index, width, height, 1, format,
                  datatype, data);
  CheckForOpenGLErrors();
  return true;
}

bool OpenGLContext::UploadTextureToArray(const cv::Mat& image,
                                         const std::string& name,
                                         const int& index) {
  cv::Mat dst = image.clone();
  // cv::flip(image, dst, 0);
  if (dst.rows % 2 != 0 || dst.cols % 2 != 0) {
    cv::resize(dst, dst,
               cv::Size(dst.cols - (dst.cols % 2), dst.rows - (dst.rows % 2)));
  }
  switch (dst.channels()) {
    case 1:
      return UploadTextureToArrayInternal(dst.data, name, dst.cols, dst.rows,
                                          GL_RED, GL_UNSIGNED_BYTE, index);
      break;
    case 3:
      return UploadTextureToArrayInternal(dst.data, name, dst.cols, dst.rows,
                                          GL_RGB, GL_UNSIGNED_BYTE, index);
      break;
    case 4:
      return UploadTextureToArrayInternal(dst.data, name, dst.cols, dst.rows,
                                          GL_RGBA, GL_UNSIGNED_BYTE, index);
      break;
    default:
      LOG(FATAL) << "UploadTextureToArray not implemented for "
                 << dst.channels() << " channel cv::Mat";
      break;
  }
}

bool OpenGLContext::UploadTextureToArray(const DepthMap& depth,
                                         const std::string& name,
                                         const int& index) {
  return UploadTextureToArrayInternal(depth.Depth().data, name, depth.Cols(),
                                      depth.Rows(), GL_RED, GL_FLOAT, index);
}

bool OpenGLContext::SetViewpoint(const Camera& camera) {
  return SetViewpoint(camera, 0.01f, 100.0f);
}

void OpenGLContext::SetViewportSize(const int& width, const int& height,
                                    const bool resize_window) {
  glfwMakeContextCurrent(window_);
  glViewport(0, 0, width, height);
  width_ = width;
  height_ = height;
  if (resize_window) {
    // glfwSetWindowSize(window_, framebuffer_size_to_screen_coords_ * width,
    //                  framebuffer_size_to_screen_coords_ * height);
  }
}

bool OpenGLContext::SetViewpoint(const Camera& camera, const float& near_clip,
                                 const float& far_clip) {
  glfwMakeContextCurrent(window_);
  CHECK(using_projection_matrix_[current_program_])
      << "Cannot call SetViewpoint on a fullscreen shader!";

  projection_ = camera.GetOpenGlMvpMatrix();

  CHECK(current_program_ >= 0) << "Did not call UseShader!";
  mvp_location_ = glGetUniformLocation(programs_[current_program_], "MVP");
  glUseProgram(programs_[current_program_]);
  width_ = camera.GetImageSize().x();
  height_ = camera.GetImageSize().y();
  // glfwSetWindowSize(window_, framebuffer_size_to_screen_coords_ * width_,
  //                  framebuffer_size_to_screen_coords_ * height_);
  glViewport(0, 0, width_, height_);

  glUniformMatrix4fv(mvp_location_, 1, GL_FALSE,
                     (const GLfloat*)projection_.data());
  CheckForOpenGLErrors();
  return true;
}

bool OpenGLContext::SetProjectionMatrix(const Eigen::Matrix4f& projection) {
  glfwMakeContextCurrent(window_);
  CHECK(using_projection_matrix_[current_program_])
      << "Cannot call SetViewpoint on a fullscreen shader!";
  projection_ = projection;

  CHECK(current_program_ >= 0) << "Did not call UseShader!";
  mvp_location_ = glGetUniformLocation(programs_[current_program_], "MVP");
  glUseProgram(programs_[current_program_]);

  glUniformMatrix4fv(mvp_location_, 1, GL_FALSE,
                     (const GLfloat*)projection_.data());
  CheckForOpenGLErrors();
  return true;
}

bool OpenGLContext::SetClearColor(const Eigen::Vector3f& color) {
  if (color.maxCoeff() > 1.0f || color.minCoeff() < 0.0f) {
    LOG(ERROR) << "Invalid color value. Expected range is 0-1.";
    return false;
  }
  clear_color_ = color;
  return true;
}

bool OpenGLContext::UseShader(const int& shader_id) {
  glfwMakeContextCurrent(window_);
  if (shader_id >= 0 && shader_id < programs_.size()) {
    CheckForOpenGLErrors();
    current_program_ = shader_id;
    glUseProgram(programs_[current_program_]);
    if (!using_projection_matrix_[shader_id]) {
      BindFullscreenTriangle();
    } else {
      BindMesh(current_mesh_);
    }
    return true;
  }
  return false;
}

void OpenGLContext::Render() {
  CHECK(is_initialized_);
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  glViewport(0, 0, width, height);
  glfwMakeContextCurrent(window_);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glClearColor(clear_color_[0], clear_color_[1], clear_color_[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  CHECK(current_program_ >= 0) << "Did not call UseShader!";
  CHECK(current_mesh_ >= 0 || !using_projection_matrix_[current_program_])
      << "Did not call BindMesh!";
  glUseProgram(programs_[current_program_]);
  if (!using_projection_matrix_[current_program_]) {
    UploadShaderUniform(1.0f, "negative");
    glBindVertexArray(vaos_[fullscreen_triangle_mesh_]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos_[fullscreen_triangle_mesh_]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos_[fullscreen_triangle_mesh_]);
    glDrawElements(GL_TRIANGLES, num_triangles_[fullscreen_triangle_mesh_] * 3,
                   GL_UNSIGNED_INT, 0);
  } else {
    glBindVertexArray(vaos_[current_mesh_]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos_[current_mesh_]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos_[current_mesh_]);
    glDrawElements(GL_TRIANGLES, num_triangles_[current_mesh_] * 3,
                   GL_UNSIGNED_INT, 0);
  }
  glfwSwapBuffers(window_);
  glfwPollEvents();
  glViewport(0, 0, width_, height_);
}

void OpenGLContext::ShowWindow() {
  glfwMakeContextCurrent(window_);
  if (!window_showing_) {
    glfwShowWindow(window_);
    glfwSetWindowSize(window_, framebuffer_size_to_screen_coords_ * width_,
                      framebuffer_size_to_screen_coords_ * height_);
    window_showing_ = true;
    CheckForOpenGLErrors();
  }
}

void OpenGLContext::HideWindow() {
  glfwMakeContextCurrent(window_);
  if (window_showing_) {
    glfwHideWindow(window_);
    window_showing_ = false;
    CheckForOpenGLErrors();
  }
}

int NumBytes(const int& datatype) {
  switch (datatype) {
    case GL_FLOAT:
      return 4;
      break;
    case GL_UNSIGNED_BYTE:
      return 1;
      break;
    case GL_INT:
      return 4;
      break;
    case GL_UNSIGNED_INT:
      return 4;
      break;
    default:
      LOG(FATAL) << "OpenGL datatype " << datatype << " not recognized.";
      return 1;
      break;
  }
}

int NumElements(const int& format) {
  switch (format) {
    case GL_RGB:
      return 3;
      break;
    case GL_RED:
      return 1;
      break;
    case GL_RGBA:
      return 4;
      break;
    case GL_RED_INTEGER:
      return 1;
      break;
    case GL_RG:
      return 2;
      break;
    default:
      LOG(FATAL) << "OpenGL format " << format << " not recognized.";
      return 1;
      break;
  }
}

void OpenGLContext::RenderToBufferInternal(void* buffer, const int& format,
                                           const int& datatype) {
  glfwMakeContextCurrent(window_);
  if (!buffers_bound_[current_program_]) {
    CHECK(CreateRenderBuffer(datatype, format));
  }
  CHECK(current_program_ >= 0) << "Did not call UseShader!";
  CHECK(current_mesh_ >= 0 || !using_projection_matrix_[current_program_])
      << "Did not call BindMesh!";
  glBindFramebuffer(GL_FRAMEBUFFER, output_buffers_[current_program_]);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (using_projection_matrix_[current_program_]) {
    projection_.row(1) *= -1;
    glCullFace(GL_FRONT);
    glUniformMatrix4fv(mvp_location_, 1, GL_FALSE,
                       (const GLfloat*)projection_.data());
    glUseProgram(programs_[current_program_]);
    glBindVertexArray(vaos_[current_mesh_]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos_[current_mesh_]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos_[current_mesh_]);
    glDrawElements(GL_TRIANGLES, num_triangles_[current_mesh_] * 3,
                   GL_UNSIGNED_INT, 0);
  } else {
    UploadShaderUniform(-1.0f, "negative");
    glCullFace(GL_FRONT);
    glUseProgram(programs_[current_program_]);
    glBindVertexArray(vaos_[fullscreen_triangle_mesh_]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos_[fullscreen_triangle_mesh_]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos_[fullscreen_triangle_mesh_]);
    glDrawElements(GL_TRIANGLES, num_triangles_[fullscreen_triangle_mesh_] * 3,
                   GL_UNSIGNED_INT, 0);
  }
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id_);
  glReadPixels(0, 0, width_, height_, format, datatype, 0);
  int pbo_size = width_ * height_ * NumBytes(datatype) * NumElements(format);
  void* ptr =
      glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, pbo_size, GL_MAP_READ_BIT);
  memcpy(buffer, ptr, pbo_size);
  glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  if (using_projection_matrix_[current_program_]) {
    projection_.row(1) *= -1;
    glUniformMatrix4fv(mvp_location_, 1, GL_FALSE,
                       (const GLfloat*)projection_.data());
    glCullFace(GL_BACK);
  } else {
    UploadShaderUniform(1.0f, "negative");
    glCullFace(GL_BACK);
  }
  glFinish();
  CheckForOpenGLErrors();
}

void OpenGLContext::RenderToImage(TriangleIdMap* array) {
  glClearColor(1, 1, 1, 1);
  RenderToBufferInternal((void*)array->data(), GL_RED_INTEGER, GL_UNSIGNED_INT);
}

void OpenGLContext::RenderToImage(DepthMap* depth) {
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  if (depth->Cols() != width_ || depth->Rows() != height_) {
    depth->Resize(height_, width_);
  }
  RenderToBufferInternal((void*)depth->MutableDepth()->data, GL_RED, GL_FLOAT);
}

void OpenGLContext::RenderToImage(cv::Mat* image) {
  // TODO(holynski): Account for non-even resolution images
  glClearColor(clear_color_[0], clear_color_[1], clear_color_[2], 0.0f);
  int intended_width = width_;
  int intended_height = height_;
  if (width_ % 2 != 0 || height_ % 2 != 0) {
    SetViewportSize(width_ - (width_ % 2), height_ - (height_ % 2));
  }
  if (image->cols != width_ || image->rows != height_) {
    *image = cv::Mat(height_, width_, image->type());
  }

  unsigned char mat_type = image->type() & CV_MAT_DEPTH_MASK;
  GLint datatype;
  switch (mat_type) {
    case CV_8U:
      datatype = GL_UNSIGNED_BYTE;
      break;
    case CV_32F:
      datatype = GL_FLOAT;
      break;
    default:
      LOG(FATAL) << "Mat type not supported.";
  }

  switch (image->channels()) {
    case 1:
      RenderToBufferInternal((void*)image->data, GL_RED, datatype);
      break;
    case 2:
      RenderToBufferInternal((void*)image->data, GL_RG, datatype);
      break;
    case 3:
      RenderToBufferInternal((void*)image->data, GL_RGB, datatype);
      break;
    case 4:
      CHECK(false);
      RenderToBufferInternal((void*)image->data, GL_RGBA, datatype);
      break;
    default:
      LOG(FATAL) << "Current rendering only supports 1-4 image channels. You "
                    "provided "
                 << image->channels() << ".";
  }
  SetViewportSize(intended_width, intended_height);
  cv::resize(*image, *image, cv::Size(width_, height_));
}

bool OpenGLContext::RenderToTexture(const std::string& name,
                                    const int shader_id) {
  glfwMakeContextCurrent(window_);
  CHECK(current_program_ >= 0) << "Did not call UseShader!";
  CHECK(current_mesh_ >= 0 || !using_projection_matrix_[current_program_])
      << "Did not call BindMesh!";

  // Create a texture
  if (textures_.count(name) < 1) {
    GLuint tex;
    glGenTextures(1, &tex);
    textures_opengl_[name] = textures_.size();
    textures_[name] = tex;
  }
  glActiveTexture(GL_TEXTURE4 + textures_opengl_[name]);
  GLuint tex = textures_[name];
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width_, height_, 0, GL_RGB,
               GL_UNSIGNED_BYTE, NULL);
  CheckForOpenGLErrors();

  // Create a framebuffer
  if (texture_framebuffers_.count(name) < 1) {
    GLuint fb = 0;
    glGenFramebuffers(1, &fb);
    texture_framebuffers_[name] = fb;
  }
  GLuint fb = texture_framebuffers_[name];
  glBindFramebuffer(GL_FRAMEBUFFER, fb);

  // Bind the texture to the framebuffer
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         tex, 0);
  GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, DrawBuffers);

  // Check that everything went smoothly
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(ERROR) << "Framebuffer could not be created.";
    return false;
  }
  CheckForOpenGLErrors();

  // Bind the framebuffer.
  glBindFramebuffer(GL_FRAMEBUFFER, fb);

  // Render as usual
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (using_projection_matrix_[current_program_]) {
    glUniformMatrix4fv(mvp_location_, 1, GL_FALSE,
                       (const GLfloat*)projection_.data());
    glCullFace(GL_BACK);
    glUseProgram(programs_[current_program_]);
    glBindVertexArray(vaos_[current_mesh_]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos_[current_mesh_]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos_[current_mesh_]);
    glDrawElements(GL_TRIANGLES, num_triangles_[current_mesh_] * 3,
                   GL_UNSIGNED_INT, 0);
  } else {
    UploadShaderUniform(-1.0f, "negative");
    glCullFace(GL_BACK);
    glUseProgram(programs_[current_program_]);
    glBindVertexArray(vaos_[fullscreen_triangle_mesh_]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos_[fullscreen_triangle_mesh_]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos_[fullscreen_triangle_mesh_]);
    glDrawElements(GL_TRIANGLES, num_triangles_[fullscreen_triangle_mesh_] * 3,
                   GL_UNSIGNED_INT, 0);
  }
  CheckForOpenGLErrors();

  // Get the location of the texture in the other shader
  glUseProgram(programs_[shader_id >= 0 ? shader_id : current_program_]);
  glActiveTexture(GL_TEXTURE4 + textures_opengl_[name]);
  glBindTexture(GL_TEXTURE_2D, tex);
  GLint texture_location = glGetUniformLocation(
      programs_[shader_id >= 0 ? shader_id : current_program_], name.c_str());
  // Upload the texture
  glUniform1i(texture_location, 4 + textures_opengl_[name]);

  // Bind the regular framebuffer again
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glUseProgram(programs_[current_program_]);

  CheckForOpenGLErrors();
  return true;
}

}  // namespace replay
