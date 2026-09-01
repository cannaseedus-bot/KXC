#version 300 es
precision highp float;

uniform sampler2D uFieldgraph;
uniform int uDimensions;
uniform int uRecord;
uniform int uColumn;

out vec4 outColor;

void main() {
    int column = clamp(uColumn, 0, uDimensions - 1);
    int record = max(uRecord, 0);
    float value = texelFetch(uFieldgraph, ivec2(column, record), 0).r;
    outColor = vec4(value, 0.0, 0.0, 1.0);
}
