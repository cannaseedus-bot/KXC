# GEMM 2×2 head-to-head — Intel HD 4600 (D3D11 FL 11_1)

`gemm_bench.cpp` — one process, one CPU reference, identical shapes. Naive + tiled(16×16)
GEMM on both D3D11 compute and OpenCL 1.2. Answers the directive:
**"commit to the OpenCL leg only if it wins."**

## Correctness FIRST (320×192×256 vs CPU f32 reference)

| kernel        | max abs err | max rel err | verdict |
|---------------|------------:|------------:|---------|
| D3D11  naive  | 6.68e-06 | 1.46e-02 | correct (rel err = near-zero-crossing metric artifact; `mad`/reassoc) |
| D3D11  tiled  | 6.68e-06 | 1.46e-02 | correct |
| OpenCL naive  | 0.00e+00 | 0.00e+00 | bit-exact (same accumulation order as CPU) |
| OpenCL tiled  | 0.00e+00 | 0.00e+00 | bit-exact |

All four numerically equivalent → **the same semantic GEMM produces equivalent results on
both backends.** (This is the SCXQ2-artifact equivalence check, not just a speed race.)

## GFLOPS (f32, reproduced across 2 runs, ±0.5)

```
1024×1024×1024        naive     tiled
  D3D11                6.8      27.3
  OpenCL              14.0      22.8

512×512×512           naive     tiled
  D3D11               14.6      32.3
  OpenCL              17.6      26.3
```

## Findings

1. **Tiling helps both** — D3D11 ~4× (6.8→27.3), OpenCL ~1.6× (14→22.8). Both benefit; D3D11 gains more from groupshared tiling.
2. **Fair comparison (tiled vs tiled): D3D11 WINS** — +20% @1024³ (27.3 vs 22.8), +23% @512³ (32.3 vs 26.3).
3. **Naive OpenCL beats naive D3D11** (14.0 vs 6.8) — OpenCL's default work distribution is better *unoptimized*; the advantage inverts once both are tiled.

## Verdict

**OpenCL does NOT win the fair (matched-optimization) comparison → do NOT commit to OpenCL
as the speed leg.** D3D11 tiled compute stays the performance primary on this rig.

This confirms the standing conclusion: **OpenCL's value is REACH, not speed** — it runs
numerically where WebGPU/Dawn is blocklisted, and bridges D3D11 zero-copy
(`cl_khr_d3d11_sharing`). Keep it as the phase-runtime's portable/reach backend; keep
D3D11 for throughput.

Build: `vcvars64 && cl /std:c++17 /EHsc /O2 gemm_bench.cpp` (OpenCL dynamically loaded from ICD).
