#version 410

out vec3 color;

uniform sampler2D input_image;

uniform sampler2D input_depth;

uniform vec3 input_position;

// The projection matrices for the input cameras.
uniform mat4 input_projection_matrix;

// Passed from vertex shader. The global 3D position of the fragment.
in vec3 fragment_position;

void main() {
  vec4 uv = input_projection_matrix * vec4(fragment_position, 1);
  if (uv.z < 0) {
    discard;
  }
  uv /= uv.z;
  uv.xy += 1.0;
  uv.xy *= 0.5;

  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
    discard;
  }

  float observed_depth = distance(fragment_position, input_position);

  float camera_depth = texture(input_depth, uv.xy).x;
  uv.y = 1.0 - uv.y;

  float depth_diff = abs(observed_depth - camera_depth);

  /*if (depth_diff > 0.1) {*/
    /*discard;*/
  /*}*/

  color = texture(input_image, uv.xy).rgb;
}
