// Shared WebGL2 HF tensor access conventions. Include this text in generated
// shader sources; it is not a standalone shader stage.
#version 300 es
precision highp float;

ivec2 hf_flat_coord(int flatIndex, ivec2 packedShape) {
    return ivec2(flatIndex % packedShape.x, flatIndex / packedShape.x);
}

float hf_get(sampler2D tensor, ivec2 packedShape, int flatIndex) {
    return texelFetch(tensor, hf_flat_coord(flatIndex, packedShape), 0).r;
}

float hf_matmul_value(sampler2D a, ivec2 aShape, sampler2D b, ivec2 bShape, ivec2 outputIndex) {
    float result = 0.0;
    for (int k = 0; k < aShape.y; ++k)
        result += texelFetch(a, ivec2(k, outputIndex.x), 0).r *
                  texelFetch(b, ivec2(outputIndex.y, k), 0).r;
    return result;
}
