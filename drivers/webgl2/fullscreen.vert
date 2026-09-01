#version 300 es
precision highp float;

out vec2 v_uv;

void main() {
    vec2 position = vec2((gl_VertexID == 2 || gl_VertexID == 1) ? 1.0 : -1.0,
                         (gl_VertexID == 2 || gl_VertexID == 3) ? 1.0 : -1.0);
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
