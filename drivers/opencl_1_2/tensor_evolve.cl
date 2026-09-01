// OpenCL C 1.2 — semantic tensor evolution.
// tensor and scratch are flat row-major buffers; shape is supplied explicitly.
__kernel void tensor_evolve(
    __global float* tensor,
    __global float* scratch,
    const int batch,
    const int sequence,
    const int hidden,
    const float dt,
    const float damping)
{
    const int idx = (int)get_global_id(0);
    const int total = batch * sequence * hidden;
    if (idx >= total) return;
    const float x = tensor[idx];
    const float dx = damping * x + scratch[idx];
    tensor[idx] = x + dt * dx;
    scratch[idx] = dx;
}
