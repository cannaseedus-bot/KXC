#pragma once

#include <vector>
#include <cstdint>

struct KVCache {
    // flattened layout: ((layer * heads + head) * max_seq + pos) * D
    std::vector<float> K; // can be int8 packed externally
    std::vector<float> V; // FP32 or FP16 promoted to float

    uint32_t layers = 0;
    uint32_t heads = 0;
    uint32_t max_seq = 0;
    uint32_t D = 0; // per-head dimension

    uint32_t current_seq = 0;

    void init(uint32_t layers_, uint32_t heads_, uint32_t max_seq_, uint32_t D_){
        layers = layers_; heads = heads_; max_seq = max_seq_; D = D_; current_seq = 0;
        size_t total = size_t(layers) * heads * max_seq * D;
        K.assign(total, 0.0f);
        V.assign(total, 0.0f);
    }

    inline uint32_t offset(uint32_t layer, uint32_t head, uint32_t pos) const{
        return ((layer * heads + head) * max_seq + pos) * D;
    }

    void append(uint32_t layer, uint32_t head, const float* k_vec, const float* v_vec){
        uint32_t pos = current_seq;
        uint32_t base = offset(layer, head, pos);
        for (uint32_t i=0;i<D;i++){
            K[base + i] = k_vec[i];
            V[base + i] = v_vec[i];
        }
    }

    // Commit slice from tempKV
    void commit_from_temp(uint32_t layer, uint32_t head, uint32_t pos_dst, const float* tempK, const float* tempV, uint32_t len){
        // pos_dst is destination position in main cache
        uint32_t base_dst = offset(layer, head, pos_dst);
        for (uint32_t i=0;i<len;i++){
            uint32_t src = i * D;
            uint32_t dst = base_dst + i * D;
            for (uint32_t d=0; d<D; d++){
                K[dst + d] = tempK[src + d];
                V[dst + d] = tempV[src + d];
            }
        }
    }
};

// Temporary KV buffer used during speculative verification
struct TempKV {
    std::vector<float> K;
    std::vector<float> V;
    uint32_t layers = 0;
    uint32_t heads = 0;
    uint32_t max_len = 0; // draft length
    uint32_t D = 0;

    void init(uint32_t layers_, uint32_t heads_, uint32_t max_len_, uint32_t D_){
        layers = layers_; heads = heads_; max_len = max_len_; D = D_;
        K.assign((size_t)layers*heads*max_len*D, 0.0f);
        V.assign((size_t)layers*heads*max_len*D, 0.0f);
    }

    inline uint32_t offset(uint32_t layer, uint32_t head, uint32_t pos) const{
        return ((layer * heads + head) * max_len + pos) * D;
    }

    void append(uint32_t layer, uint32_t head, uint32_t pos, const float* k_vec, const float* v_vec){
        uint32_t base = offset(layer, head, pos);
        for (uint32_t i=0;i<D;i++){ K[base + i] = k_vec[i]; V[base + i] = v_vec[i]; }
    }

    void reset(){ std::fill(K.begin(), K.end(), 0.0f); std::fill(V.begin(), V.end(), 0.0f); }
};

// commit a prefix from TempKV into main KVCache
inline void commit_prefix_from_temp(KVCache &main, const TempKV &tmp, uint32_t accepted, uint32_t dst_pos_start){
    for (uint32_t layer=0; layer<tmp.layers; ++layer){
        for (uint32_t head=0; head<tmp.heads; ++head){
            for (uint32_t i=0;i<accepted;++i){
                uint32_t src_off = tmp.offset(layer, head, i);
                main.commit_from_temp(layer, head, dst_pos_start + i, &tmp.K[src_off], &tmp.V[src_off], 1);
            }
        }
    }
}
