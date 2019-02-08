#version 410

out vec3 color;

// The input camera images
uniform sampler2DArray input_images;

// The depth maps rasterized for the input cameras using the current mesh
uniform sampler2DArray input_depths;

// The camera centers for the input images
uniform vec3 input_positions[100];

// The projection matrices for the input cameras.
uniform mat4 input_projection_matrices[100];

// The position from which we're currently rendering.
uniform vec3 virtual_camera_position;

// The number of input cameras.
uniform int num_cameras; 
// Passed from vertex shader. The global 3D position of the fragment.
in vec3 fragment_position;

void main() {
  int camera1 = -1;
  int camera2 = -1;
  int camera3 = -1;
  float weight1 = 9999999.0;
  float weight2 = 9999999.0;
  float weight3 = 9999999.0;
  vec4 uv1 = vec4(0, 0, 0, 0);
  vec4 uv2 = vec4(0, 0, 0, 0);
  vec4 uv3 = vec4(0, 0, 0, 0);
  for (int camera = 0; camera < 100; camera++) {
    vec3 camera_ray = normalize(fragment_position - virtual_camera_position);
    vec3 projector_ray = normalize(fragment_position - input_positions[camera]);
    float distance_cost =
        max(1 - (distance(fragment_position, virtual_camera_position) /
                 distance(fragment_position, input_positions[camera])),
            0);
    float angle_cost = acos(dot(camera_ray, projector_ray)) * 180.0 / 3.14159;
    float weight = 0.9 * angle_cost + 0.1 * distance_cost;
    vec4 uv = input_projection_matrices[camera] * vec4(fragment_position, 1);
    if (uv.z < 0) {
      continue;
    }
    uv /= uv.z;
    uv.xy += 1.0;
    uv.xy *= 0.5;

    if (uv.x <= 0.01 || uv.y <= 0.01 || uv.x >= 0.99 || uv.y >= 0.99) {
      continue;
    }

    uv.z = camera;
    float observed_depth = distance(fragment_position, input_positions[camera]);

    float camera_depth = texture(input_depths, uv.xyz).x;
    uv.y = 1.0 - uv.y;

    float depth_diff = abs(observed_depth - camera_depth);

    //if (depth_diff > 0.1) continue;

    if (weight < weight1) {
      weight3 = weight2;
      weight2 = weight1;
      weight1 = weight;
      camera3 = camera2;
      camera2 = camera1;
      camera1 = camera;
      uv3 = uv2;
      uv2 = uv1;
      uv1 = uv;
    } else if (weight < weight2) {
      weight3 = weight2;
      weight2 = weight;
      camera3 = camera2;
      camera2 = camera;
      uv3 = uv2;
      uv2 = uv;
    } else if (weight < weight3) {
      weight3 = weight;
      camera3 = camera;
      uv3 = uv;
    }
  }

  float sigma = weight3 * 0.33;
  weight1 = exp(-weight1 / sigma);
  weight2 = exp(-weight2 / sigma);
  weight3 = exp(-weight3 / sigma);

  //if (weight1 + weight2 + weight3 - 0.01 < 0.0) discard;
  float sum_weight = weight1 + weight2 + weight3;
  weight1 = weight1 / sum_weight;
  weight2 = weight2 / sum_weight;
  weight3 = weight3 / sum_weight;

  vec3 color1 = texture(input_images, uv1.xyz).rgb;
  /*vec3 color2 = weight2 * texture(input_images, uv2.xyz).rgb;*/
  /*vec3 color3 = weight3 * texture(input_images, uv3.xyz).rgb;*/
  color = color1;
}
