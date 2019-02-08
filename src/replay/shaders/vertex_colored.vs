#version 410

out vec3 frag_rgb;

uniform mat4 MVP;

in vec3 vert;
in vec3 rgb;

void main() {
  gl_Position = MVP * vec4(vert, 1.0);
  frag_rgb = rgb;
}
