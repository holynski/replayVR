#version 410

out vec3 color;

uniform sampler2D input_image;
uniform vec3 input_position;
uniform mat4 input_projection_matrix;
uniform mat4 rotation; 
uniform mat4 intrinsics; 

uniform vec2 image_size;

void main() {

  vec2 pixel_coord = (gl_FragCoord.xy / image_size);
  vec3 camera_ray = (inverse(intrinsics) * vec4(pixel_coord, 1, 1)).xyz;
  vec3 world_ray = (transpose(rotation) * vec4(camera_ray,1)).xyz;

  world_ray = normalize(world_ray);
  vec2 polar;
  polar.x = acos(world_ray.x);
  polar.y = atan(world_ray.y,world_ray.z);

  polar.y += (3.14159 * 0.50);
  /*polar.y -= (3.14159 * 0.40);*/
  /*polar.x -= (3.14159 * 0.1);*/

  polar /= 3.14159;
  vec2 uv = polar;

  /*if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {*/
    /*discard;*/
  /*}*/

  if (uv.y > 1.0) {
    uv.y -= 1;
    uv.x = 1-uv.x;
  }
  if (uv.x > 1.0) {
    uv.x -= 1;
    uv.y = 1-uv.y;
  }
  if (uv.x < 0) {
    uv.x += 1;
    uv.y = 1-uv.y;
  }
  if (uv.y < 0) {
    uv.y += 1;
    uv.x = 1-uv.x;
  }

  color = texture(input_image, uv.xy).rgb;
  /*color = vec3(uv.xy, 0);*/
  /*color = projection; */
}
