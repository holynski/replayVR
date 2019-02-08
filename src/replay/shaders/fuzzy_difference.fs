#version 410

out vec3 color;

uniform sampler2D image1;
uniform sampler2D image2;
uniform int window_size;
uniform vec2 image_size;

void main() {
  ivec2 center_pixel = ivec2(gl_FragCoord.x,  gl_FragCoord.y);
  ivec2 half_patch = ivec2(window_size / 2, window_size / 2);
  ivec2 patch_min = max(center_pixel - half_patch, ivec2(0, 0));
  ivec2 patch_max =
      min(center_pixel + half_patch, ivec2(image_size.x - 1, image_size.y - 1));
  vec3 image1_pixel = texelFetch(image1, center_pixel, 0).rgb;
  vec3 min_difference = vec3(1, 1, 1);
  for (int y = patch_min.y; y <= patch_max.y; y++) {
    for (int x = patch_min.x; x <= patch_max.x; x++) {
      vec3 image2_pixel = texelFetch(image2, ivec2(x, y), 0).rgb;
      vec3 difference = image1_pixel - image2_pixel;
      if (difference.x < 0 || difference.y < 0 || difference.z < 0) {
        difference = vec3(1, 1, 1);
      }
      min_difference = min(min_difference, difference);
    }
  }
  if (min_difference.x == 1 && min_difference.y == 1 && min_difference.z == 1) {
    min_difference = vec3(0, 0, 0);
  }
  color = min_difference;
  /*color = image1_pixel;*/
}
