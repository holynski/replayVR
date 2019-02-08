#version 410

out vec3 color;

// The depth maps rasterized for the input cameras using the current mesh
uniform sampler2D min_composite; 

// The input camera images
uniform sampler2D input_image;

// The depth maps rasterized for the input cameras using the current mesh
uniform sampler2D input_depth;

// The camera centers for the input images
uniform vec3 position;

// The projection matrices for the input cameras.
uniform mat4 projection_matrix;

// The position from which we're currently rendering.
uniform vec3 virtual_camera_position;
// The number of input cameras.
// Passed from vertex shader. The global 3D position of the fragment.
in vec3 fragment_position;

void main() {
  color = texelFetch(min_composite, ivec2(gl_FragCoord.xy), 0).rgb;

  // Populate the projection matrix
  vec4 uv = projection_matrix * vec4(fragment_position, 1);
  if (uv.z < 0) {
    return;
  }
  uv /= uv.z;
  uv.xy += 1.0;
  uv.xy *= 0.5;

  if (uv.x < 0 || uv.y < 0 || uv.x > 1.0 || uv.y > 1.0) {
    return;
  }

  float observed_depth = distance(fragment_position, position);

  uv.y = 1.0 - uv.y;
  float camera_depth = texture(input_depth, uv.xy).x;
  if (observed_depth > 0 && camera_depth > 0 &&
      (observed_depth / camera_depth > 1.005 ||
       camera_depth / observed_depth > 1.005)) {
    return;
  }

  vec3 obs = texture(input_image, uv.xy).rgb;

  if (length(obs) == 0) {
    return;
  }
  color = min(color.rgb, obs.rgb);
}
