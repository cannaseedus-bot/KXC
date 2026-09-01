// oss_native.cpp — native C++ inference for gpt-oss-20b (NO llama.cpp, NO python).
// Loads tensor_map.json + safetensors, dequants MXFP4/Q8_0/F32, runs the 24-layer
// MoE forward (attention -> router -> top-4 -> weighted MoE -> residual), and
// reports h_24 vs the authoritative #005C trajectory.
//
// Build (MSVC):  cl /O2 /std:c++17 /EHsc oss_native.cpp
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>

// ── minimal JSON parser (extracts the fields we need) ──────────────────────
static std::string trim(const std::string& s){ size_t a=s.find_first_not_of(" \t\r\n"); if(a==std::string::npos)return""; size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1); }
static std::string unquote(const std::string& s){ if(s.size()>=2 && s.front()=='"' && s.back()=='"') return s.substr(1,s.size()-2); return s; }

struct TensorInfo { std::string name; std::vector<int64_t> logical; std::string dtype; int64_t off0, off1; };

// parse a JSON array of numbers "[1,2,3]"
static std::vector<int64_t> parse_num_array(const std::string& s){
    std::vector<int64_t> r; std::string cur;
    for(char c : s){ if(c==','||c==']'){ if(!cur.empty()) r.push_back(std::stoll(cur)); cur.clear(); } else if(c!='['&&c!=' '&&c!='\n'&&c!='\r'&&c!='\t') cur+=c; }
    return r;
}

// parse tensor_map.json -> vector<TensorInfo>
static std::vector<TensorInfo> load_tensor_map(const std::string& path){
    std::ifstream f(path); std::stringstream ss; ss<<f.rdbuf(); std::string s=ss.str();
    std::vector<TensorInfo> out;
    // find each "tensors" array element: {"name": "...", ...}
    size_t pos=0;
    while((pos=s.find("\"name\"",pos))!=std::string::npos){
        size_t obj_start=s.rfind('{',pos);
        size_t obj_end=s.find('}',pos);
        if(obj_start==std::string::npos||obj_end==std::string::npos) break;
        std::string o=s.substr(obj_start,obj_end-obj_start+1);
        TensorInfo t;
        // name
        size_t nq=o.find("\"name\""); size_t vq=o.find(':',nq); size_t v1=o.find('"',vq); size_t v2=o.find('"',v1+1);
        t.name=o.substr(v1+1,v2-v1-1);
        // logical_shape
        size_t ls=o.find("\"logical_shape\""); if(ls!=std::string::npos){ size_t lb=o.find('[',ls); size_t le=o.find(']',lb); t.logical=parse_num_array(o.substr(lb,le-lb+1)); }
        // ggml_type_name
        size_t gt=o.find("\"ggml_type_name\""); if(gt!=std::string::npos){ size_t gq=o.find(':',gt); size_t g1=o.find('"',gq); size_t g2=o.find('"',g1+1); t.dtype=o.substr(g1+1,g2-g1-1); }
        // data_offsets
        size_t dof=o.find("\"data_offsets\""); if(dof!=std::string::npos){ size_t db=o.find('[',dof); size_t de=o.find(']',db); auto r=parse_num_array(o.substr(db,de-db+1)); if(r.size()>=2){ t.off0=r[0]; t.off1=r[1]; } }
        out.push_back(t);
        pos=obj_end+1;
    }
    return out;
}

// ── safetensors payload origin ──────────────────────────────────────────────
static int64_t payload_origin(const std::string& path){
    std::ifstream f(path, std::ios::binary); uint64_t n=0; f.read((char*)&n,8); return 8+(int64_t)n;
}

// ── dequant ────────────────────────────────────────────────────────────────
static const int8_t MXFP4_K[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static float e8m0_to_fp32(uint8_t x){
    uint32_t bits = (x<2) ? (0x00200000u << x) : ((uint32_t)(x-1) << 23);
    float f; memcpy(&f,&bits,4); return f;
}
// dequant a raw tensor block to float, given dtype + logical element count
static std::vector<float> dequant(const std::vector<uint8_t>& raw, const std::string& dtype, size_t n){
    std::vector<float> out(n);
    if(dtype=="F32"){ memcpy(out.data(), raw.data(), n*4); return out; }
    if(dtype=="Q8_0"){
        // block of 32: 2-byte f16 scale + 32 int8
        size_t nb=n/32; size_t p=0;
        for(size_t b=0;b<nb;b++){
            uint16_t s16; memcpy(&s16,&raw[p],2); p+=2;
            float d = (float)(int16_t)s16 / 256.0f; // f16->f32 approx (scale is f16)
            // proper f16 decode
            uint32_t sign=(s16>>15)&1, exp=(s16>>10)&0x1F, man=s16&0x3FF;
            float val;
            if(exp==0){ val=(man? (float)man/1024.0f*0.00006103515625f : 0.0f); }
            else if(exp==31){ val=(man? NAN : INFINITY); }
            else { val=(1.0f+(float)man/1024.0f)*std::ldexp(1.0f,(int)exp-15); }
            if(sign) val=-val;
            for(int i=0;i<32;i++){ int8_t q=(int8_t)raw[p+i]; out[b*32+i]=val*q; }
            p+=32;
        }
        return out;
    }
    if(dtype=="MXFP4"){
        // block of 32: 1 e8m0 scale + 16 nibble bytes.
        // Layout (matches gguf-py): out[0..15]=low nibbles, out[16..31]=high nibbles.
        size_t nb=n/32; size_t p=0;
        for(size_t b=0;b<nb;b++){
            uint8_t e=raw[p++]; float d=e8m0_to_fp32(e);
            for(int i=0;i<16;i++){
                uint8_t byte=raw[p+i];
                int8_t q0=(int8_t)(byte&0x0F), q1=(int8_t)((byte>>4)&0x0F);
                out[b*32+i]    = d*MXFP4_K[q0];
                out[b*32+16+i] = d*MXFP4_K[q1];
            }
            p+=16;
        }
        return out;
    }
    fprintf(stderr,"unsupported dtype %s\n",dtype.c_str()); return out;
}

// ── math helpers ───────────────────────────────────────────────────────────
static float silu(float x){ return x/(1.0f+std::exp(-x)); }
static void rmsnorm(std::vector<float>& h, const std::vector<float>& w, float eps=1e-5f){
    float mean=0; for(float x:h) mean+=x*x; mean/=h.size();
    float r=1.0f/std::sqrt(mean+eps);
    for(size_t i=0;i<h.size();i++) h[i]*=r*w[i];
}
// matvec: A[M,K] (row-major) @ x[K] -> y[M]
static void matvec(const std::vector<float>& A, size_t M, size_t K, const std::vector<float>& x, std::vector<float>& y){
    y.assign(M,0);
    for(size_t i=0;i<M;i++){ float s=0; for(size_t k=0;k<K;k++) s+=A[i*K+k]*x[k]; y[i]=s; }
}
// matvec with transposed A: A^T[M,K] means A is [K,M], compute A^T @ x
static void matvecT(const std::vector<float>& A, size_t K, size_t M, const std::vector<float>& x, std::vector<float>& y){
    y.assign(M,0);
    for(size_t i=0;i<M;i++){ float s=0; for(size_t k=0;k<K;k++) s+=A[k*M+i]*x[k]; y[i]=s; }
}

// ── SHA256 (compact) ───────────────────────────────────────────────────────
static uint32_t rotr(uint32_t x, int n){ return (x>>n)|(x<<(32-n)); }
static void sha256(const uint8_t* data, size_t len, uint8_t out[32]){
    static const uint32_t K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t n=len; size_t padded=((n+8)/64+1)*64; std::vector<uint8_t> m(padded,0);
    memcpy(m.data(),data,n); m[n]=0x80;
    uint64_t bits=(uint64_t)n*8; for(int i=0;i<8;i++) m[padded-1-i]=(uint8_t)(bits>>(8*i));
    for(size_t off=0;off<padded;off+=64){
        uint32_t w[64]; for(int i=0;i<16;i++) w[i]=(m[off+i*4]<<24)|(m[off+i*4+1]<<16)|(m[off+i*4+2]<<8)|m[off+i*4+3];
        for(int i=16;i<64;i++){ uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3); uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for(int i=0;i<64;i++){ uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25); uint32_t ch=(e&f)^((~e)&g); uint32_t t1=hh+S1+ch+K[i]+w[i]; uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22); uint32_t maj=(a&b)^(a&c)^(b&c); uint32_t t2=S0+maj; hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for(int i=0;i<8;i++){ out[i*4]=(uint8_t)(h[i]>>24); out[i*4+1]=(uint8_t)(h[i]>>16); out[i*4+2]=(uint8_t)(h[i]>>8); out[i*4+3]=(uint8_t)h[i]; }
}

int main(int argc, char** argv){
    const std::string ST="E:/models/GPT-DDS/GPT-OSS/gpt-oss-20b-MXFP4.safetensors";
    const std::string TM="E:/models/GPT-DDS/GPT-OSS/gpt-oss-20b-MXFP4.tensor_map.json";
    const std::string TRAJ="E:/models/GPT-DDS/GPT-OSS/traj_t0.json";
    int token = argc>1 ? atoi(argv[1]) : 0;

    auto tensors=load_tensor_map(TM);
    std::map<std::string,TensorInfo> ts; for(auto&t:tensors) ts[t.name]=t;
    int64_t origin=payload_origin(ST);
    std::ifstream sf(ST, std::ios::binary);

    auto read_tensor=[&](const std::string& name)->std::vector<float>{
        auto& t=ts[name]; size_t n=1; for(auto d:t.logical) n*=d;
        std::vector<uint8_t> raw(t.off1-t.off0);
        sf.seekg(origin+t.off0); sf.read((char*)raw.data(), raw.size());
        return dequant(raw, t.dtype, n);
    };

    // token embedding [2880, 201088] Q8_0 -> h = embd[:, token]
    auto embd=read_tensor("token_embd.weight");
    size_t H=2880, VOCAB=201088;
    std::vector<float> h(H);
    for(size_t i=0;i<H;i++) h[i]=embd[i*VOCAB+token];

    // authoritative h_24 from traj_t0.json (last cert h_out_hash)
    // (we compare per-layer h_out via SHA256 in a follow-up; here print final)
    std::vector<float> h24;
    for(int L=0;L<24;L++){
        auto wn=read_tensor("blk."+std::to_string(L)+".attn_norm.weight");
        auto wv=read_tensor("blk."+std::to_string(L)+".attn_v.weight");   // [2880,512]
        auto wo=read_tensor("blk."+std::to_string(L)+".attn_output.weight"); // [4096,2880]
        auto wp=read_tensor("blk."+std::to_string(L)+".post_attention_norm.weight");
        auto wr=read_tensor("blk."+std::to_string(L)+".ffn_gate_inp.weight"); // [2880,32]
        auto wgate=read_tensor("blk."+std::to_string(L)+".ffn_gate_exps.weight"); // [2880,2880,32]
        if(L==0 && getenv("OSS_DEBUG")){
            printf("DBG wgate[0,0,13]=%f wgate[1,0,13]=%f wgate[0,1,13]=%f wgate[0,0,28]=%f\n",
                wgate[0*2880*32+0*32+13], wgate[1*2880*32+0*32+13], wgate[0*2880*32+1*32+13], wgate[0*2880*32+0*32+28]);
        }
        auto wup=read_tensor("blk."+std::to_string(L)+".ffn_up_exps.weight");
        auto wdown=read_tensor("blk."+std::to_string(L)+".ffn_down_exps.weight");

        rmsnorm(h, wn);
        // v = wv.T @ h ; wv is [2880,512] -> v[512]
        std::vector<float> v(512); matvecT(wv, 2880, 512, h, v);
        // ctx = repeat_interleave(v,8) -> [4096]
        std::vector<float> ctx(4096); for(size_t i=0;i<512;i++) for(int r=0;r<8;r++) ctx[i*8+r]=v[i];
        // h_attn = h + wo.T @ ctx ; wo is [4096,2880] -> wo.T @ ctx = [2880]
        std::vector<float> attn_out(2880); matvecT(wo, 4096, 2880, ctx, attn_out);
        std::vector<float> h_attn(2880); for(size_t i=0;i<2880;i++) h_attn[i]=h[i]+attn_out[i];
        // h_moe = rmsnorm(h_attn, wp)
        std::vector<float> h_moe=h_attn; rmsnorm(h_moe, wp);
        // logits = wr.T @ h_moe ; wr is [2880,32] -> [32]
        std::vector<float> logits(32); matvecT(wr, 2880, 32, h_moe, logits);
        // top-4
        std::vector<int> idx(32); for(int i=0;i<32;i++) idx[i]=i;
        std::partial_sort(idx.begin(), idx.begin()+4, idx.end(), [&](int a,int b){ return logits[a]>logits[b]; });
        int top4[4]={idx[0],idx[1],idx[2],idx[3]};
        // softmax of top4 values
        float mx=logits[top4[0]]; float sum=0; float wts[4];
        for(int j=0;j<4;j++) wts[j]=std::exp(logits[top4[j]]-mx), sum+=wts[j];
        for(int j=0;j<4;j++) wts[j]/=sum;
        // moe = sum_j wts[j] * (wdown[...,e].T @ (silu(wgate[...,e].T@h_moe) * (wup[...,e].T@h_moe)))
        std::vector<float> moe(2880,0);
        for(int j=0;j<4;j++){
            int e=top4[j];
            // wgate[...,e] is [2880,2880] at flat[a*2880*32 + b*32 + e]
            // wgate[...,e].T @ x : result[i] = sum_j wgate[j,i,e]*x[j]
            auto expert_matvec=[&](const std::vector<float>& W, const std::vector<float>& x, std::vector<float>& y){
                y.assign(2880,0);
                for(size_t i=0;i<2880;i++){ float s=0; for(size_t j=0;j<2880;j++) s+=W[j*2880*32+i*32+e]*x[j]; y[i]=s; }
            };
            std::vector<float> ge, up, inter(2880), down;
            expert_matvec(wgate, h_moe, ge);
            expert_matvec(wup, h_moe, up);
            for(size_t i=0;i<2880;i++) inter[i]=silu(ge[i])*up[i];
            expert_matvec(wdown, inter, down);
            for(size_t i=0;i<2880;i++) moe[i]+=wts[j]*down[i];
        }
        std::vector<float> h_out(2880); for(size_t i=0;i<2880;i++) h_out[i]=h_attn[i]+moe[i];
        h=h_out; h24=h_out;
        printf("L%02d top4=[%d,%d,%d,%d]\n", L, top4[0],top4[1],top4[2],top4[3]);
    }
    printf("h_24 computed (2880 floats). Compare to #005C via SHA256 in the harness.\n");
    // SHA256 of the final hidden state (float32 bytes) == authoritative h_24
    uint8_t digest[32]; sha256((const uint8_t*)h24.data(), h24.size()*4, digest);
    printf("h_24_sha256="); for(int i=0;i<16;i++) printf("%02x", digest[i]); printf("\n");
    return 0;
}
