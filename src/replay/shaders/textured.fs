#version 410

out vec3 color;
in vec2 frag_uv;
uniform sampler2D image;

void main() {
    color = texture(image, frag_uv.xyz).rgb;
}
