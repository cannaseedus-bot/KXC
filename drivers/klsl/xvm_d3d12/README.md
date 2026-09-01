# KLSL XVM/D3D12 Target

This is the XVM/D3D12 shader target carried by the KLSL driver.

```text
shaders/                     XVM HLSL and WGSL shader assets
../bin/kxc.exe               KXC semantic compiler
../bin/xvm_d12.exe           XVM D3D12 runtime
../bin/*_smoke.exe           runtime smoke tests
```

The target is intentionally separate from the native KLSLC HLSL/XVM target:

- XVM/D3D12 assets are D3D12-only.
- WGSL files here belong to the XVM/WebGPU-side asset history and are not the
  WebGL2 GLSL ES 3.00 backend.
- OpenCL remains OpenCL C 1.2 only.
- On Intel HD 4600, this target is not selected because D3D12 feature level is
  unavailable; use the bounded OpenCL 1.2 or D3D11 CS5 paths instead.
