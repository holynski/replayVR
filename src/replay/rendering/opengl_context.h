#ifndef REPLAY_OPENGL_RENDERER_H_
#define REPLAY_OPENGL_RENDERER_H_

#if defined(__APPLE__)
#define GLFW_INCLUDE_GLCOREARB
#elif defined(__linux__) || defined(__unix__) || defined(__posix__)
#include <GL/glew.h>
#else
//#include <windows.h>
#include <GL/glew.h>
#include <GL/gl.h>

#endif  // __APPLE__
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>

#include "replay/depth_map/depth_map.h"
#include "replay/mesh/mesh.h"
#include "replay/mesh/triangle_id_map.h"
#include "replay/third_party/theia/sfm/camera/camera.h"
#include "replay/util/row_array.h"

namespace replay {

// Creates an OpenGL composite projection-extrinsics matrix from a Camera
// that can be passed to the UploadShaderUniform function.
Eigen::Matrix4f GetOpenGLMatrix(const theia::Camera& camera);

// The OpenGLContext class is a wrapper for shader rendering via OpenGL.
// Multiple instances of this class cannot exist on different threads safely.
// Each instance of this class has a single render window.
//
// Example usage:
//    OpenGLContext renderer;
//    renderer.Initialize();
//    renderer.CompileAndLinkShaders(vertex, fragment, &id);
//    renderer.UseShader(id);
//    renderer.UploadMesh(mesh);
//    renderer.SetViewpoint(camera);
//    renderer.Render() *or* renderer.RenderToImage(&image);
class OpenGLContext {
 public:
  // Creates an OpenGL renderer object. The OpenGL context is not created until
  // the Initialize() function is called.
  OpenGLContext();

  // Destroys all the shaders and meshes that have been uploaded to the GPU, and
  // if there are not any other OpenGLContext objects currently instantiated,
  // destroys the OpenGL context.
  ~OpenGLContext();

  // Sets up the OpenGL context. Must be called before using anything else in
  // this class.
  // Returns true if context creation was successful, and otherwise false.
  bool Initialize();

  // Checks if the UploadMesh() function has been called with a valid mesh, and
  // the mesh has been uploaded to the GPU.
  bool HasMesh() const;

  // Checks if the Initialize() function has been successfully called.
  bool IsInitialized() const;

  // Shows the rendering window.
  void ShowWindow();

  // Hides the rendering window.
  void HideWindow();

  // Uploads a mesh to the GPU. Must be called before any Render*() call. If a
  // mesh already exists, it will be replaced.
  // Returns true if the mesh was sucessfully uploaded and bound, and false
  // otherwise.
  bool UploadMesh(const Mesh& mesh);

  // Sets the callback function to be called when the mouse moves.
  // The function signature should be:
  //
  // void Callback(double x, double y);
  bool SetMousePositionCallback(void (*callback)(double, double));

  // Sets the callback function to be called when a mouse button is pressed on
  // the render window. The function signature should be:
  //
  // void Callback(int button, int action, int modifier);
  //
  // - The button values are those defined by GLFW, found here:
  //   http://www.glfw.org/docs/latest/group__buttons.html
  // - The action value will be either GLFW_PRESS or GLFW_RELEASE.
  // - The modifier value defines whether any modifier keys (CTRL, ALT, SHIFT)
  //   were pressed while the button was pressed. The values are defined here:
  //   http://www.glfw.org/docs/latest/group__mods.html
  bool SetMouseClickCallback(void (*callback)(int, int, int));

  // Sets the callback function to be called when a keyboard key is pressed on
  // the render window. The function signature should be:
  //
  // void Callback(int key, int action, int modifier);
  //
  // - The key values are those defined by GLFW, found here:
  //   http://www.glfw.org/docs/latest/group__keys.html
  // - The action value will be either GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT.
  // - The modifier value defines whether any modifier keys (CTRL, ALT, SHIFT)
  //   were pressed while the main key was pressed. The values are defined here:
  //   http://www.glfw.org/docs/latest/group__mods.html
  bool SetKeyboardCallback(void (*callback)(int, int, int));

  // Returns the last recorded mouse position (X,Y).
  Eigen::Vector2d GetMousePosition() const;

  // Reads a shader source from the filepath and stores the result into a
  // string. This is a handy way to allow writing shaders to files for complex
  // shaders.
  static bool ReadShaderFromFile(const std::string& filename,
                                 std::string* shader_src);

  // Creates and links a vertex/fragment shader pair.
  // If successful, the output value shader_id will be set to the newly created
  // shader's internal ID. This ID can be used in calls to UseShader().
  // Returns true if successful, false otherwise.
  bool CompileAndLinkShaders(const std::string& vertex,
                             const std::string& fragment, int* shader_id);

  // Compiles a "full-screen" shader where each fragment is active in the
  // fragment shader. This allows you to perform operations on the pixels of
  // images easily, such as applying blurs or computing gradients.
  //
  // Interally, this function will destroy any existing geometric data
  // (i.e. any uploaded meshes) by rendering one single huge triangle that the
  // entire screen will fit on to. This has been shown to be more effective than
  // using a quad.
  bool CompileFullScreenShader(const std::string& fragment, int* shader_id);

  // Activates a particular shader by its ID value. ID values are returned by
  // the CompileAndLinkShaders function. Must be called before any Render*()
  // call.
  // Returns true if successful, false otherwise.
  bool UseShader(const int& shader_id);

  // Sets the OpenGL virtual camera to be in the same position and orientation
  // as the provided camera.
  // If provided, the last two parameters define the distance from the camera
  // center to the near and far clipping planes. Otherwise, they are set to the
  // default values of 0.01 and 100, respectively.
  // The render viewport (size of the output rendered image) is also resized to
  // the size of the theia::Camera image. If rendering to the window, an attempt
  // is made to resize the window to the
  // viewport size. If this is not possible, due to OS window manager
  // constraints, the window remains at its current size.
  // Returns true if the viewpoint was set successfully, false otherwise.
  bool SetViewpoint(const theia::Camera& camera);
  bool SetViewpoint(const theia::Camera& camera, const float& near,
                    const float& far);
  
  // If not using the above SetViewpoint functions, you may instead use these two 
  // functions, setting the MVP matrix manually, and adjusting the viewport (and window) 
  // size. 
  bool SetProjectionMatrix(const Eigen::Matrix4f& projection);
  void SetViewportSize(const int& width, const int& height, const bool resize_window = true);

  // Uploads values to the shader. It is expected that these values correspond
  // in type and name to uniforms defined in the shader.
  // In specific, for each of the following types, the shader must contain the
  // corresponding declaration before the shader's main() function:
  //    Eigen::Matrix4f => uniform mat4 <name>;
  //    Eigen::Vector3f => uniform vec3 <name>;
  //    Eigen::Vector2f => uniform vec2 <name>;
  //    float => uniform float <name>;
  //    cv::Mat => uniform sampler2D <name>;
  //    DepthMap => uniform sampler2D <name>;
  //
  // Texture uploading functions will return false if the texture could not be
  // bound. This may be the case if too many textures are uploaded or if the
  // texture size is too large.
  void UploadShaderUniform(const int& val, const std::string& name);
  void UploadShaderUniform(const float& val, const std::string& name);
  void UploadShaderUniform(const Eigen::Vector2f& val, const std::string& name);
  void UploadShaderUniform(const Eigen::Vector3f& val, const std::string& name);
  void UploadShaderUniform(const Eigen::Matrix4f& val, const std::string& name);
  void UploadShaderUniform(const std::vector<Eigen::Matrix4f>& val,
                           const std::string& name);
  void UploadShaderUniform(const std::vector<Eigen::Vector3f>& val,
                           const std::string& name);
  bool UploadTexture(const cv::Mat& image, const std::string& name);
  bool UploadTexture(const DepthMap& depth, const std::string& name);
  GLuint GetTextureId(const std::string& name) const;

  // Uploads a single texture image to a texture array at a given index.
  // AllocateTextureArray must be called before any calls to
  // UploadTextureToArray.
  //
  // The shader must contain "uniform sampler2DArray <name>".
  //
  // Returns false if the image could not be uploaded. This may happen if the
  // the index specified is too large for the allocated space or the image size
  // is too large.
  bool UploadTextureToArray(const cv::Mat& image, const std::string& name,
                            const int& index);
  bool UploadTextureToArray(const DepthMap& depth_map, const std::string& name,
                            const int& index);

  // Allocates the memory for a texture array.
  // Width, height, and channels define the size of each image. The values must
  // be the same across all items stored in the texture array.
  // For a texture array containing cv::Mat elements, these values are
  // stored in image.Cols(), image.Rows(), and image.Channels(), respectively.
  // For a texture array containing replay::DepthMap elements, the width and
  // height are stored in depth.Cols(), depth.Rows(), and channels is equal to
  // 1.
  //
  // Num_elements define the number of images which will be held in the array.
  //
  // The shader must contain "uniform sampler2DArray <name>".
  //
  // The final argument will define whether the textures uploaded to this array
  // will be compressed. If supported by the driver, this will cause texture
  // upload to be slower, but textures will be much smaller in size.
  bool AllocateTextureArray(const std::string& name, const int& width,
                            const int& height, const int& channels,
                            const int& num_elements,
                            const bool compressed = false);

  // Renders a frame using the given shader, mesh, and uniforms to an image in
  // memory.
  // Depending on your shader's output datatype, you should call the function
  // with a container supporting the correct datatype. Calling even once with
  // the incorrect datatype will create an internal buffer with the requested
  // datatype, and all future calls to RenderToImage will be affected.
  // If you call RenderImage with a:  || Your fragment shader must contain:
  //   TriangeIdMap                   ||  out uint color;
  //   cv::Mat                        ||  out [float/vec3/vec4] color;
  //   DepthMap                       ||  out float color;
  void RenderToImage(TriangleIdMap* array);
  void RenderToImage(cv::Mat* image);
  void RenderToImage(DepthMap* depth);

  // Renders a frame to the window. Will render to the window even if it is
  // hidden, and the image will continue to be displayed until the next call to
  // Render().
  void Render();

 protected:
  static int instantiated_renderers_;
  static std::unordered_map<GLFWwindow*, OpenGLContext*> window_to_renderer_;
  int current_program_;
  std::vector<GLuint> programs_;
  std::vector<GLuint> fragment_shaders_;
  std::vector<GLuint> vertex_shaders_;
  std::vector<GLuint> output_buffers_;
  std::vector<bool> using_projection_matrix_;
  GLuint pbo_id_;
  std::vector<bool> buffers_bound_;
  GLint mvp_location_;
  GLuint vao_;
  GLuint vbo_;
  GLuint ebo_;
  GLuint uvbo_;
  GLFWwindow* window_;
  int width_;
  int height_;
  std::unordered_map<std::string, GLuint> textures_;
  std::unordered_map<std::string, int> textures_opengl_;
  Eigen::Matrix4f projection_;
  int num_triangles_;
  bool is_initialized_;
  bool has_mesh_;
  bool window_showing_;
  static float framebuffer_size_to_screen_coords_;
  Eigen::Vector2d mouse_position_;
  bool BindFullscreenTriangle();
  void RenderToBufferInternal(void* buffer, const int& format,
                              const int& datatype);
  bool UploadTextureInternal(void* data, const int& width, const int& height,
                             const int& format, const int& datatype,
                             const int& internal_format,
                             const std::string& name);
  bool UploadTextureToArrayInternal(const void* data, const std::string& name,
                                    const int& width, const int& height,
                                    const int& format, const int& index);
  bool CreateRenderBuffer(const int& datatype, const int& format);
  void DestroyContext();
  static void KeyboardCallback(GLFWwindow* window, int key, int scancode,
                               int action, int mods);
  static void MouseButtonCallback(GLFWwindow* window, int button, int action,
                                  int mods);
  static void MousePosCallback(GLFWwindow* window, double x, double y);
  void (*Keyboard_)(int, int, int);
  void (*MouseMove_)(double, double);
  void (*MouseButton_)(int, int, int);
};

}  // namespace replay

#endif  // REPLAY_OPENGL_H_
