// xcfe_matmul_test.c — prove XCFE actually COMPUTES MUL_MAT (not just registers).
// Runs the same ggml_mul_mat graph on (a) ggml's CPU backend = ground truth, and (b) the XCFE
// backend, then asserts the outputs match. This validates ggml_backend_xcfe_graph_compute against
// ggml's canonical MUL_MAT semantics — the CPU baseline the GPU glyph kernel later replaces.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-xcfe.h"
#include <stdio.h>
#include <math.h>

#define K 64  // shared dim
#define N 32  // A cols  -> C rows
#define M 8   // B cols  -> C cols  (a representative layer-tile shape for the DirectML path)

static void fill(float * p, int n, unsigned seed) {
    for (int i = 0; i < n; ++i) { seed = seed * 1103515245u + 12345u; p[i] = ((seed >> 16) & 0x7fff) / 32768.0f - 0.5f; }
}

static int run_on(ggml_backend_t backend, const float * Ad, const float * Bd, float * Cout) {
    struct ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, /*no_alloc*/ true };
    struct ggml_context * ctx = ggml_init(p);

    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);   // -> [N, M]

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(A, Ad, 0, ggml_nbytes(A));
    ggml_backend_tensor_set(B, Bd, 0, ggml_nbytes(B));

    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, C);
    enum ggml_status st = ggml_backend_graph_compute(backend, g);

    ggml_backend_tensor_get(C, Cout, 0, ggml_nbytes(C));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return st == GGML_STATUS_SUCCESS ? 0 : 1;
}

int main(void) {
    float A[K * N], B[K * M], Ccpu[N * M], Cxcfe[N * M];
    fill(A, K * N, 1u);
    fill(B, K * M, 2u);

    ggml_backend_t cpu  = ggml_backend_cpu_init();
    ggml_backend_t xcfe = ggml_backend_xcfe_init();
    if (!cpu || !xcfe)               { printf("backend init failed\n"); return 3; }
    if (run_on(cpu,  A, B, Ccpu))    { printf("cpu compute failed\n");  return 3; }
    if (run_on(xcfe, A, B, Cxcfe))   { printf("xcfe compute failed\n"); return 3; }

    double maxerr = 0.0, maxval = 0.0;
    for (int i = 0; i < N * M; ++i) {
        double e = fabs((double) Ccpu[i] - (double) Cxcfe[i]);
        if (e > maxerr) maxerr = e;
        if (fabs((double) Ccpu[i]) > maxval) maxval = fabs((double) Ccpu[i]);
    }
    double nrm = maxval > 0 ? maxerr / maxval : maxerr;
    printf("MUL_MAT [K=%d,N=%d,M=%d]  (%d elems)\n", K, N, M, N * M);
    printf("  sample: C[0] cpu=% .6f xcfe=% .6f   C[%d] cpu=% .6f xcfe=% .6f\n",
           Ccpu[0], Cxcfe[0], N * M - 1, Ccpu[N * M - 1], Cxcfe[N * M - 1]);
    printf("  max abs err = %.3e   (scale-normalized %.3e)\n", maxerr, nrm);
    int ok = nrm < 1e-3;   // DirectML f32 vs ggml CPU f32
    printf("XCFE computes MUL_MAT, matches ggml CPU: %s\n", ok ? "YES" : "NO");

    ggml_backend_free(cpu);
    ggml_backend_free(xcfe);
    return ok ? 0 : 4;
}
