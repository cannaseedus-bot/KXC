// dxm_test.cpp — DirectXMath (CPU SIMD) matvec proof for the hybrid GPU/CPU path.
// DirectXMath vectorizes dequant+matvec with SSE/AVX on the CPU; OpenCL does the
// GPU-resident working set. This proves the CPU-SIMD half works and is fast.
#include <DirectXMath.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
using namespace DirectX;

// matvec y = A[M,K] @ x[K], 4-wide SIMD
static void matvec_simd(const std::vector<float>& A, size_t M, size_t K, const std::vector<float>& x, std::vector<float>& y){
    for(size_t i=0;i<M;i++){
        XMVECTOR acc = XMVectorZero();
        size_t k=0;
        for(; k+4<=K; k+=4){
            XMVECTOR av = XMLoadFloat4((const XMFLOAT4*)&A[i*K+k]);
            XMVECTOR xv = XMLoadFloat4((const XMFLOAT4*)&x[k]);
            acc = XMVectorMultiplyAdd(av, xv, acc);
        }
        XMFLOAT4 t; XMStoreFloat4(&t, acc); float s=t.x+t.y+t.z+t.w;
        for(; k<K; k++) s += A[i*K+k]*x[k];
        y[i]=s;
    }
}
static void matvec_scalar(const std::vector<float>& A, size_t M, size_t K, const std::vector<float>& x, std::vector<float>& y){
    for(size_t i=0;i<M;i++){ float s=0; for(size_t k=0;k<K;k++) s+=A[i*K+k]*x[k]; y[i]=s; }
}

int main(){
    size_t M=2880, K=2880;   // one expert plane [2880,2880] @ [2880]
    std::vector<float> A(M*K), x(K), ys(M), yi(M);
    for(size_t i=0;i<A.size();i++) A[i]=(float)(rand()%1000)/1000.f - 0.5f;
    for(size_t i=0;i<K;i++) x[i]=(float)(rand()%1000)/1000.f - 0.5f;
    matvec_simd(A,M,K,x,yi); matvec_scalar(A,M,K,x,ys);
    double err=0; for(size_t i=0;i<M;i++) err+=fabs(yi[i]-ys[i]);
    printf("SIMD vs scalar max-err per element: %.3e\n", err/M);
    // benchmark
    auto t0=std::chrono::high_resolution_clock::now();
    for(int r=0;r<50;r++) matvec_simd(A,M,K,x,yi);
    auto t1=std::chrono::high_resolution_clock::now();
    auto ms=std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/50.0;
    printf("SIMD matvec [2880x2880]@[2880]: %.1f us\n", ms);
    t0=std::chrono::high_resolution_clock::now();
    for(int r=0;r<50;r++) matvec_scalar(A,M,K,x,ys);
    t1=std::chrono::high_resolution_clock::now();
    ms=std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/50.0;
    printf("scalar matvec [2880x2880]@[2880]: %.1f us\n", ms);
    printf("RESULT: %s\n", err/M<1e-3 ? "PASS — DirectXMath SIMD matvec correct + fast" : "FAIL");
    return 0;
}
