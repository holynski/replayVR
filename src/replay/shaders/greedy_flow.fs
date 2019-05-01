#version 410

out vec2 flow; 

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
  float min_difference = 3;
  flow = vec2(0,0);
  for (int y = patch_min.y; y <= patch_max.y; y++) {
    for (int x = patch_min.x; x <= patch_max.x; x++) {
      vec3 image2_pixel = texelFetch(image2, ivec2(x, y), 0).rgb;
      vec3 difference = image1_pixel - image2_pixel;
      if (difference.x < 0 || difference.y < 0 || difference.z < 0) {
        difference = vec3(1, 1, 1);
      }
      float diff_length = length(difference);
      if (diff_length < min_difference) {
        min_difference = diff_length;
        flow = vec2(x,y) - vec2(center_pixel);
      }
    }
  } 
}
