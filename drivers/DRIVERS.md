{
  "drivers": "consolidated GPU/driver runtime + sources (single location, no longer scattered across /dist)",
  "gpu_runtime": {
    "GLSL_Server.exe": "OpenGL 4.3 compute server (HD 4600-compatible) — from dist/kuhul-es/bin",
    "neural_layer.glsl": "GLSL neural layer kernel",
    "server.glsl.json": "GLSL server config",
    "kuhul-es.js": "kuhul-es gateway",
    "xvm_d12.dll": "D3D12 driver (FL12+ only; HD 4600 can't run) — from dist/xvm-d3d12/drivers",
    "xvm_d12_host.exe": "D3D12 host",
    "scx_runtime.exe": "MoE router runtime (D3D12 device init fails on FL11.x) — from dist/micronaut-v2",
    "scx_shaders/": "scx_runtime shaders",
    "kxc.exe": "KXC shader compiler — from dist/xvm-d3d12/build-ninja3/kxc",
    "compiler-kxc/": "KXC compiler runtime",
    "cs5_shaders/": "D3D11 cs_5_0 CSOs (HD-4600-ready) — from dist/v3.5.0-WebX/shaders/cs5"
  },
  "model_server": {
    "llama-server.exe": "WORKING model server (loads llama-server-impl.dll). Replaces buggy khanary-server.exe (removed from server bat)."
  },
  "native_drivers": {
    "khanary_driver.dll": "in-process TaskEngine + DAG + provider dispatch",
    "gl_infer_driver.dll": "OpenGL inference driver",
    "kuhul_engine_driver.dll": "kuhul_engine driver",
    "qwen_infer_driver.dll": "qwen inference driver",
    "native_glyph_engine.dll": "glyph engine",
    "xcfe_gl_ops.dll": "GL ops for XCFE",
    "json_runtime_lib.dll": "json runtime lib"
  },
  "sources": "*.cpp/*.h + build_drivers.bat (MSVC BuildTools 2022)",
  "note": "GPU path on HD 4600 = GLSL_Server (OpenGL 4.3) + cs5_shaders. D3D12 pieces (xvm_d12, scx_runtime) are FL12+ only."
}
