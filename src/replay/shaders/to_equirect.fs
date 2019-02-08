#version 410

out vec3 color;

uniform sampler2D input_image;
uniform vec3 input_position;
uniform mat4 input_projection_matrix;
uniform mat4 rotation; 
uniform mat4 intrinsics; 

uniform vec2 image_size;

void main() {
  vec2 polar = (gl_FragCoord.xy / image_size);
  polar.y *= 3.14159;
  polar.y -= 3.14159 * 0.5;
  polar.x *= 3.14159;
  /*polar.y += (3.14159 * 0.40);*/
  /*polar.x += (3.14159 * 0.1);*/
  vec3 angle_ray = 
                        vec3(
                        cos(polar.x),
                        sin(polar.x)*sin(polar.y),
                        sin(polar.x)*cos(polar.y)
);
  
  vec3 projection = (rotation * vec4(angle_ray,1)).xyz;
  projection /= projection.z;
  vec3 uv = (intrinsics * vec4(projection,1)).xyz;
  
  /*[>vec4 uv = input_projection_matrix * vec4(angle_ray + input_position, 1);<]*/
  if (uv.z < 0) {
    discard;
  }
  /*[>uv.xy += 1.0;<]*/
  /*[>uv.xy *= 0.5;<]*/

  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
    discard;
  }

  color = texture(input_image, uv.xy).rgb;
  /*color = vec3(uv.xy, 0);*/
  /*color = projection; */
}
