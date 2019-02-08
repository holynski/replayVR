#version 410

out vec3 color;

// The input camera images
uniform sampler2DArray input_images;

// The depth maps rasterized for the input cameras using the current mesh
uniform sampler2DArray input_depths;

// The camera centers for the input images
uniform vec3 input_positions[100];

// The projection matrices for the input cameras.
uniform mat4 input_projection_matrices[200];

// The position from which we're currently rendering.
uniform vec3 virtual_camera_position;

uniform int cam_id;

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
  float count = 0.0;
  color = vec3(1,1,1);
  vec4 base_uv = input_projection_matrices[cam_id] * vec4(fragment_position, 1);
  base_uv /= base_uv.z;
  base_uv.xy += 1.0;
  base_uv.xy *= 0.5;
  base_uv.z = cam_id;
  base_uv.y = 1.0 - base_uv.y;
  
  
  vec3 base_color = texture(input_images, base_uv.xyz).rgb;
  base_color = pow(base_color, 2.2);

  for (int camera = 0; camera < 200; camera++) {
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

    /*if (depth_diff > 0.1) continue;*/
  
    vec3 obs = texture(input_images, uv.xyz).rgb;
    obs = pow(obs, 2.2);
    color = min(color,obs);
    /*count = count + 1.0;*/
  }

  color = base_color - color;
  color = max(color, vec3(0,0,0));
}
