// ============================================================================
// bimodal_attention.hlsl - A1 ⊗ A2 Multi-Manifold Fusion
// ============================================================================

// A1: Flat Euclidean Attention (QKV)
RWStructuredBuffer<float> flatAttention : register(u0);

// A2: Curved Riemannian Attention (Manifold)
RWStructuredBuffer<float> curvedAttention : register(u1);

// Output: Tensor Product Fusion
RWStructuredBuffer<float> fusedOutput : register(u2);

cbuffer DispatchInfo : register(b0) {
    uint batch_size;
    uint num_heads;
    uint dim_flat;
    uint dim_curved;
};

// ⟁ LAW A: COMPUTE PHASE - Bimodal Fusion
// Fused(i, j) = A1(i) ⊗ A2(j) preserves both independent geometric signatures.

[numthreads(8, 8, 8)]
void BimodalFusion(uint3 id : SV_DispatchThreadID) {
    uint b = id.x; // Batch
    uint h = id.y; // Head
    uint idx_fused = id.z; // Fused Dimension Index
    
    if (b >= batch_size || h >= num_heads || idx_fused >= (dim_flat * dim_curved)) return;
    
    // Decompose the fused index into A1 and A2 indices (Kronecker product mapping)
    uint i = idx_fused / dim_curved;
    uint j = idx_fused % dim_curved;
    
    // Calculate global indices for reading flat and curved buffers
    uint global_flat_idx = (b * num_heads * dim_flat) + (h * dim_flat) + i;
    uint global_curved_idx = (b * num_heads * dim_curved) + (h * dim_curved) + j;
    uint global_fused_idx = (b * num_heads * dim_flat * dim_curved) + (h * dim_flat * dim_curved) + idx_fused;
    
    // Read orthogonal geometry components
    float val_flat = flatAttention[global_flat_idx];
    float val_curved = curvedAttention[global_curved_idx];
    
    // Tensor Product Fusion
    // Multiplication preserves the provenance of both the semantic similarity (flat)
    // and the structural locality (curved).
    fusedOutput[global_fused_idx] = val_flat * val_curved;
}
