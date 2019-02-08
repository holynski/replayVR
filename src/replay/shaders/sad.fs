#version 410

out float color;

uniform sampler2D image1;
uniform sampler2D image2;
uniform sampler2D mask1;
uniform sampler2D mask2;
uniform int window_size;

void main() {

  ivec2 c = ivec2(gl_FragCoord.xy);
  int hw = window_size / 2;
  ivec2 wmin = max(c - hw, ivec2(0,0));
  ivec2 wmax = min(c + hw, textureSize(image1, 0));

  int count = 0;
  float sum = 0;
  for (int x = wmin.x; x < wmax.x; x++) {
    for (int y = wmin.y; y < wmax.y; y++) {
      if (texelFetch(mask1, ivec2(x,y), 0).r == 0) {
        continue;
      }
      if (texelFetch(mask2, ivec2(x,y), 0).r == 0) {
        continue;
      }
      sum += distance(
          texelFetch(image1, ivec2(x,y), 0),
          texelFetch(image2, ivec2(x,y), 0));
      count += 1;
    }
  }

  if (count == 0) {
    color = 1000;
    return;
  }

  color = sum / count;
}
