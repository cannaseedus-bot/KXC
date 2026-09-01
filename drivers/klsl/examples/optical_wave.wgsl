// optical_wave.wgsl — WGSL port of optical_wave.klsl
// Spherical-harmonic wave propagation compute shader.
// Each thread: one SH lattice node, 9 coefficients (l=0..2).
// Phase-rotates each band by dt * freq[l], then energy-normalises.

struct WaveConst {
    node_count : u32,
    sh_bands   : u32,   // 3 bands → 9 coefficients
    dt         : f32,
    _pad       : f32,
};

@group(0) @binding(0) var<storage, read>       sh_in : array<f32>;
@group(0) @binding(1) var<storage, read_write> sh_out: array<f32>;
@group(0) @binding(2) var<uniform>             cb    : WaveConst;

@compute @workgroup_size(256, 1, 1)
fn wave_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nid = gid.x;
    if (nid >= cb.node_count) { return; }

    let base = nid * 9u;

    let c0 = sh_in[base + 0u];
    let c1 = sh_in[base + 1u];
    let c2 = sh_in[base + 2u];
    let c3 = sh_in[base + 3u];
    let c4 = sh_in[base + 4u];
    let c5 = sh_in[base + 5u];
    let c6 = sh_in[base + 6u];
    let c7 = sh_in[base + 7u];
    let c8 = sh_in[base + 8u];

    // Band 0 (l=0): freq = 1.0
    let r0 = c0 * cos(cb.dt * 1.0);

    // Band 1 (l=1): freq = 2.0
    let cs1 = cos(cb.dt * 2.0);
    let r1 = c1 * cs1;
    let r2 = c2 * cs1;
    let r3 = c3 * cs1;

    // Band 2 (l=2): freq = 3.0
    let cs2 = cos(cb.dt * 3.0);
    let r4 = c4 * cs2;
    let r5 = c5 * cs2;
    let r6 = c6 * cs2;
    let r7 = c7 * cs2;
    let r8 = c8 * cs2;

    // Energy-conserving normalise (unit L2 norm)
    let norm2 = r0*r0 + r1*r1 + r2*r2 + r3*r3
              + r4*r4 + r5*r5 + r6*r6 + r7*r7 + r8*r8;
    let inv_n = select(1.0, inverseSqrt(norm2), norm2 > 1e-8);

    sh_out[base + 0u] = r0 * inv_n;
    sh_out[base + 1u] = r1 * inv_n;
    sh_out[base + 2u] = r2 * inv_n;
    sh_out[base + 3u] = r3 * inv_n;
    sh_out[base + 4u] = r4 * inv_n;
    sh_out[base + 5u] = r5 * inv_n;
    sh_out[base + 6u] = r6 * inv_n;
    sh_out[base + 7u] = r7 * inv_n;
    sh_out[base + 8u] = r8 * inv_n;
}
