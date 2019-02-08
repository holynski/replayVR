#version 410

out vec2 frag_uv;

uniform mat4 MVP;

in vec3 vert;
in vec2 uv;

void main() {
  gl_Position = MVP * vec4(vert, 1.0);
  fragment_position = vert;
}
