// Placeholder for wave-optimized fused QKV + attention path.
// Intended for hardware where caps.wave_ops == true.
// Note: full implementation should be compiled with SM6/DXC path.

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    // Intentionally empty placeholder.
}
