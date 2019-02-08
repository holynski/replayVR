#version 410

out vec3 fragment_position;

uniform mat4 MVP;

in vec3 vert;

void main() {
  gl_Position = MVP * vec4(vert, 1.0);
  fragment_position = vert;
}
