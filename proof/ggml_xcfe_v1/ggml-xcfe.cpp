// ggml-xcfe.cpp — KHANARY-native ggml backend (the KHΛNARY slot in the ggml registry).
//
// A real, distinct, registering backend that BUILDS with only the CPU backend present. It mirrors
// the minimal-real shape of ggml-blas: buffers delegate to the CPU backend, and `supports_op`
// currently returns false so the scheduler never routes ops here (everything falls back to CPU).
// `graph_compute` is therefore a no-op success with ZERO external dependencies. The K'UHUL
// glyph-lowering compute (MUL_MAT -> KHANARY glyph kernels) replaces the stub later.
#include "ggml-impl.h"
#include "ggml-xcfe.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"   // ggml_backend_cpu_buffer_type / ggml_backend_buft_is_host
#ifdef _WIN32
#include <windows.h>
#endif
#include <cstdio>
#include <cstdlib>

// The DirectML GEMM lives in dml_gemm.dll (proof/kuhul_matmul_tick_v1 / scratch/dml). We load it at
// runtime so ggml-xcfe carries NO DirectML link dependency and degrades gracefully to the CPU
// baseline when the dll / GPU isn't available. Override the dll path with $KHANARY_DML_GEMM.
//   int dml_gemm_bt_f32(const float* A, const float* B, float* C, uint M,uint N,uint K)
//     C[M,N] = A[M,K] @ B^T   (B row-major [N,K])
typedef int (*xcfe_dml_gemm_bt_fn)(const float *, const float *, float *, unsigned, unsigned, unsigned);

static xcfe_dml_gemm_bt_fn xcfe_dml_gemm(void) {
    static xcfe_dml_gemm_bt_fn fn = NULL;
    static bool tried = false;
    if (tried) { return fn; }
    tried = true;
#ifdef _WIN32
    const char * path = getenv("KHANARY_DML_GEMM");
    HMODULE h = LoadLibraryA(path ? path : "dml_gemm.dll");
    if (h) { fn = (xcfe_dml_gemm_bt_fn) GetProcAddress(h, "dml_gemm_bt_f32"); }
    fprintf(stderr, "[ggml-xcfe] MUL_MAT path: %s\n", fn ? "DirectML (GPU)" : "CPU baseline (dml_gemm.dll unavailable)");
#endif
    return fn;
}

// ---------------------------------------------------------------------------- backend interface

static const char * ggml_backend_xcfe_get_name(ggml_backend_t backend) {
    return "XCFE";
    GGML_UNUSED(backend);
}

static void ggml_backend_xcfe_free(ggml_backend_t backend) {
    delete backend;
}

// Baseline MUL_MAT: dst[n,m] = sum_k a[k,n] * b[k,m]  (ggml mul_mat semantics: dot of columns).
// a: ne0=K, ne1=N (contiguous F32); b: ne0=K, ne1=M; dst: ne0=N, ne1=M. This is the CPU FALLBACK —
// the KHANARY glyph / DirectML GEMM (proven in proof/kuhul_matmul_tick_v1) swaps in here for the
// GPU path; the op claimed by supports_op stays the same.
static void ggml_backend_xcfe_mul_mat(const struct ggml_tensor * dst) {
    const struct ggml_tensor * a = dst->src[0];
    const struct ggml_tensor * b = dst->src[1];

    const int64_t K = a->ne[0];
    const int64_t N = a->ne[1];
    const int64_t M = b->ne[1];

    const float * A = (const float *) a->data;
    const float * B = (const float *) b->data;
    float       * D = (float *)       dst->data;

    // GPU path: dst[n,m] = sum_k a[k,n]*b[k,m]  ==  C[M,N] = b_rm[M,K] @ a_rm[N,K]^T, which is exactly
    // dml_gemm_bt_f32(A_arg=b, B_arg=a, C=dst, M, N, K). Succeeds -> done; any failure -> CPU baseline.
    xcfe_dml_gemm_bt_fn dml = xcfe_dml_gemm();
    if (dml && dml(B, A, D, (unsigned) M, (unsigned) N, (unsigned) K) == 0) {
        return;
    }

    // CPU baseline (fallback)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double s = 0.0;
            const float * ac = A + n * K;
            const float * bc = B + m * K;
            for (int64_t k = 0; k < K; ++k) {
                s += (double) ac[k] * (double) bc[k];
            }
            D[m * N + n] = (float) s;
        }
    }
}

static enum ggml_status ggml_backend_xcfe_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // XCFE claims MUL_MAT (see supports_op) and computes it here on the CPU baseline. Metadata ops
    // are no-ops. The scheduler only routes claimed ops here, so nothing else should appear.
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_xcfe_mul_mat(node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: XCFE received unclaimed op %s\n", __func__, ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static struct ggml_backend_i xcfe_backend_i = {
    /* .get_name                = */ ggml_backend_xcfe_get_name,
    /* .free                    = */ ggml_backend_xcfe_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xcfe_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xcfe_guid(void) {
    // distinct KHANARY guid (spells nothing; just unique vs other backends)
    static ggml_guid guid = { 0x4b, 0x48, 0x41, 0x4e, 0x41, 0x52, 0x59, 0x00, 0x78, 0x63, 0x66, 0x65, 0x11, 0x22, 0x33, 0x44 };
    return &guid;
}

ggml_backend_t ggml_backend_xcfe_init(void) {
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xcfe_guid(),
        /* .iface   = */ xcfe_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xcfe_reg(), 0),
        /* .context = */ nullptr,
    };
    return backend;
}

bool ggml_backend_is_xcfe(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xcfe_guid());
}

// ---------------------------------------------------------------------------- device interface

static const char * ggml_backend_xcfe_device_get_name(ggml_backend_dev_t dev) {
    return "XCFE";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_xcfe_device_get_description(ggml_backend_dev_t dev) {
    return "KHANARY XCFE (K'UHUL glyph backend, stub: ops fall back to CPU)";
    GGML_UNUSED(dev);
}

static void ggml_backend_xcfe_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xcfe_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_xcfe_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xcfe_device_get_name(dev);
    props->description = ggml_backend_xcfe_device_get_description(dev);
    props->type        = ggml_backend_xcfe_device_get_type(dev);
    ggml_backend_xcfe_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xcfe_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xcfe_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xcfe_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();   // delegate storage to the CPU backend
    GGML_UNUSED(dev);
}

static bool ggml_backend_xcfe_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        // metadata / view ops: no compute, always "supported"
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        // XCFE claims plain 2D F32 contiguous matmul (the KHANARY GEMM shape). Batched / non-F32 /
        // non-contiguous cases are declined -> the scheduler leaves them on CPU.
        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * a = op->src[0];
            const struct ggml_tensor * b = op->src[1];
            return a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(a) && ggml_is_contiguous(b) &&
                   a->ne[2] == 1 && a->ne[3] == 1 &&
                   b->ne[2] == 1 && b->ne[3] == 1;
        }

        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_xcfe_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xcfe_device_i = {
    /* .get_name             = */ ggml_backend_xcfe_device_get_name,
    /* .get_description      = */ ggml_backend_xcfe_device_get_description,
    /* .get_memory           = */ ggml_backend_xcfe_device_get_memory,
    /* .get_type             = */ ggml_backend_xcfe_device_get_type,
    /* .get_props            = */ ggml_backend_xcfe_device_get_props,
    /* .init_backend         = */ ggml_backend_xcfe_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xcfe_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_xcfe_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xcfe_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------- reg interface

static const char * ggml_backend_xcfe_reg_get_name(ggml_backend_reg_t reg) {
    return "XCFE";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_xcfe_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xcfe_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xcfe_device = {
        /* .iface   = */ ggml_backend_xcfe_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xcfe_device;
    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_xcfe_reg_i = {
    /* .get_name         = */ ggml_backend_xcfe_reg_get_name,
    /* .get_device_count = */ ggml_backend_xcfe_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xcfe_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_xcfe_reg(void) {
    static struct ggml_backend_reg ggml_backend_xcfe_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xcfe_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xcfe_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xcfe_reg)
