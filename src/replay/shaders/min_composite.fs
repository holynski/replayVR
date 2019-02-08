#version 410

out vec3 color;

// The input camera images
uniform sampler2DArray input_images;

// The depth maps rasterized for the input cameras using the current mesh
uniform sampler2DArray input_depths;

// The camera centers for the input images
uniform sampler2D position;

// The projection matrices for the input cameras.
uniform sampler2D projection_matrix;

// The position from which we're currently rendering.
uniform vec3 virtual_camera_position; 
// The number of input cameras.
// Passed from vertex shader. The global 3D position of the fragment.
in vec3 fragment_position;

void main() {
  color = vec3(1, 1, 1);

  for (int cam = 0; cam < NUM_CAMERAS; cam++) {
    // Populate the projection matrix
    mat4 projection;
    for (int i = 0; i < 4; i++) {
      for (int j = cam*4; j < 4 + cam*4; j++) {
        projection[i][j%4] = texelFetch(projection_matrix, ivec2(i, j), 0).r;
      }
    }

    // Populate the position
    vec3 pos;
    for (int i = 0; i < 3; i++) {
      pos[i] = texelFetch(position, ivec2(cam, i), 0).r;
    }

    vec4 uv = projection * vec4(fragment_position, 1);
    if (uv.z < 0) {
      continue;
    }
    uv /= uv.z;
    uv.xy += 1.0;
    uv.xy *= 0.5;

    if (uv.x < 0 || uv.y < 0 || uv.x > 1.0 || uv.y > 1.0) {
      continue; 
    }

    float observed_depth = distance(fragment_position, pos);

    uv.y = 1.0 - uv.y;
    float camera_depth = texture(input_depths, vec3(uv.xy, cam)).x;
    if (observed_depth > 0 && camera_depth > 0 &&
        (observed_depth / camera_depth > 1.005 ||
         camera_depth / observed_depth > 1.005)) {
      continue;
    }

    vec3 obs = texture(input_images, vec3(uv.xy, cam)).rgb;

    if (length(obs) == 0) {
      continue;
    }
    color = min(color.rgb, obs.rgb);
  }
}
