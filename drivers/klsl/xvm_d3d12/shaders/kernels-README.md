# kernels/ — GPU Compute Shaders & XVM-D3D12 Build

## Status: Disconnected (Future Phase)

GPU compute kernels (WGSL/HLSL) for MoE inference operations, plus the XVM-D3D12 native build system.

## WGSL Shaders (Root)

| File | Purpose |
|------|---------|
| `compression-ops.wgsl` | Weight compression/decompression on GPU |
| `expert-forward.wgsl` | MoE expert forward pass |
| `int4-kernel-intel-igpu.wgsl` | INT4 quantized kernel for Intel iGPU |
| `sparse-routing.wgsl` | Sparse expert routing dispatch |

## xvm-d3d12/ — Native D3D12 Build

CMake-based C++ project that compiles XVM bytecode executors and the `atomizer` tool.

```
xvm-d3d12/
  CMakeLists.txt          Build configuration
  asx_manifest.json       ASX module manifest
  src/                    C++ source (xvm_core.h, d3d12_engine.h, HLSL shaders)
  build/                  MSVC build output (atomizer.exe, vcxproj files)
  models/                 xjson model definitions (atomizer.xjson, micronaut_demo.xjson)
```

## Build

Requires Visual Studio 2022+ with Windows SDK and CMake:
```bash
cd kernels/xvm-d3d12
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

## Integration Status

**Not referenced** by active EBPD system. These kernels target Phase 6+ GPU compute acceleration.
