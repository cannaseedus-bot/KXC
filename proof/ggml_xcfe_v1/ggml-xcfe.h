#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// KHANARY-native ggml backend (the KHΛNARY slot in the ggml backend registry).
//
// MILESTONE (this file): the backend is a real, distinct, registering target that BUILDS — it
// de-orphans `ggml-xcfe` (which upstream ships as a byte-copy of ggml-webgpu). `supports_op`
// currently returns false, so the scheduler routes nothing here and every op falls back to CPU.
// The K'UHUL glyph-lowering compute (MUL_MAT -> KHANARY glyph kernels) lands in `graph_compute`
// as the follow-on; buffers delegate to the CPU backend for now.

GGML_BACKEND_API ggml_backend_t     ggml_backend_xcfe_init(void);
GGML_BACKEND_API bool               ggml_backend_is_xcfe(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xcfe_reg(void);

#ifdef  __cplusplus
}
#endif
