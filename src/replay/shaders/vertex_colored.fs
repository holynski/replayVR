#version 410

out vec3 color;
in vec3 frag_rgb;

void main() {
    color = frag_rgb.bgr;
}
