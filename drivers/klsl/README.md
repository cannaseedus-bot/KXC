# KLSL Driver

This directory is the project-local KLSL shader driver.

```text
drivers/klsl/
  bin/klslc.exe       native KLSL compiler driver
  bin/kxc.exe         XVM semantic compiler driver
  bin/xvm_d12.exe     XVM D3D12 runtime
  klslc.cmd           local command shim
  klsl_compiler.*     compiler source
  klsl_opcodes.h      XVM opcode definitions
  examples/           KLSL shader examples
  xvm_d3d12/          XVM/D3D12 shader target and runtime assets
```

Usage from the repository root:

```powershell
drivers\klsl\klslc.cmd drivers\klsl\examples\neural_layer.klsl `
  drivers\klsl\examples\neural_layer.hlsl
```

`klslc.exe` currently emits HLSL. The CLI accepts `--xvm`, but the checked-in
compiler implementation does not yet populate the XVM result. OpenCL 1.2 and
WebGL2 remain runtime backend drivers selected by
`programs/klsl.targets.json`, `programs/opencl.kernel.json`, and
`programs/webgl2.kernel.json`.
