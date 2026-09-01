// fabric_test.cpp — parallel DirectXMath linalg across a CPU thread cluster.
// Each thread runs SIMD matvec on a partition of the MoE expert planes.
// This is the CPU "fabric": N threads x DirectXMath SIMD, in sequence with OpenCL.
#include <DirectXMath.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
using namespace DirectX;

static void matvec_simd_rows(const std::vector<float>& A, size_t M, size_t K, const std::vector<float>& x, std::vector<float>& y, size_t r0, size_t r1){
    for(size_t i=r0;i<r1;i++){
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

int main(){
    size_t M=2880, K=2880;
    std::vector<float> A(M*K), x(K), y(M);
    for(size_t i=0;i<A.size();i++) A[i]=(float)(rand()%1000)/1000.f-0.5f;
    for(size_t i=0;i<K;i++) x[i]=(float)(rand()%1000)/1000.f-0.5f;

    // reference single-thread SIMD
    matvec_simd_rows(A,M,K,x,y,0,M);
    std::vector<float> ref=y;

    auto bench=[&](int nthreads){
        std::vector<std::thread> ts;
        size_t per=M/nthreads;
        auto t0=std::chrono::high_resolution_clock::now();
        for(int r=0;r<20;r++){
            ts.clear();
            for(int t=0;t<nthreads;t++){
                size_t r0=t*per, r1=(t==nthreads-1)?M:(t+1)*per;
                ts.emplace_back(matvec_simd_rows, std::cref(A), M, K, std::cref(x), std::ref(y), r0, r1);
            }
            for(auto&t:ts) t.join();
        }
        auto t1=std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/20.0;
    };

    printf("CPU thread cluster (Haswell, %d logical cores):\n", std::thread::hardware_concurrency());
    double base=0;
    for(int nt : {1,2,4,8}){
        double us=bench(nt);
        if(nt==1) base=us;
        printf("  %d threads: %.0f us/op  speedup=%.2fx\n", nt, us, base/us);
    }
    // correctness
    double err=0; for(size_t i=0;i<M;i++) err+=fabs(y[i]-ref[i]);
    printf("err vs single-thread: %.3e\n", err/M);
    printf("RESULT: %s\n", err/M<1e-3 ? "PASS — parallel DirectXMath fabric works" : "FAIL");
    return 0;
}
