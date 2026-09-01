/**
 * @file fabric_kernel_minimal.hlsl
 * @brief Minimal Fabric compute kernel for attention execution
 * 
 * Single kernel that processes all Fabric instructions for a node.
 * Compact and efficient enough for real-time inference.
 * 
 * Execution model:
 *   - One thread group per node
 *   - Threads process instructions sequentially (per PC)
 *   - State shared via global memory
 */

StructuredBuffer<uint4> Instructions : register(t0);  // Packed FabricInstruction
RWStructuredBuffer<float> State : register(u0);       // Global state

cbuffer ExecutionParams : register(b0)
{
    uint node_count;
    uint instr_per_node;
    uint state_stride;
    uint _pad;
};

/**
 * @brief Execute single node's instruction stream
 * 
 * Dispatch: (node_count / 64 + 1) thread groups
 * Each group has 64 threads
 */
[numthreads(64, 1, 1)]
void FabricMain(uint3 id : SV_DispatchThreadID, uint tid : SV_GroupIndex)
{
    uint node_id = id.x;
    if (node_id >= node_count)
        return;

    // Each node has fixed instruction stream
    uint pc = node_id * instr_per_node;
    uint state_base = node_id * state_stride;

    // Execute instruction sequence
    for (uint i = 0; i < instr_per_node; i++)
    {
        uint4 instr_packed = Instructions[pc + i];

        // Unpack instruction
        uint opcode = instr_packed.x & 0xFFFF;
        uint flags = (instr_packed.x >> 16) & 0xFFFF;
        uint op_a = instr_packed.y;
        uint op_b = instr_packed.z;
        uint op_c = instr_packed.w;

        // Execute operation
        switch (opcode)
        {
        case 5:  // OP_ATTENTION
        {
            // Simplified attention: q * k * v
            // op_a = query offset, op_b = key offset, op_c = value offset
            
            float q = State[state_base + op_a];
            float k = State[state_base + op_b];
            float v = State[state_base + op_c];

            // Compute attention score
            float score = q * k;  // Simplified (real: dot product)
            float output = score * v;

            // Store result back to state
            State[state_base + op_c] = output;
            break;
        }

        case 6:  // OP_SOFTMAX
        {
            // Simplified softmax: exp(x) / sum(exp(x))
            // op_a = input buffer, op_b = output buffer, op_c = dimension
            
            float max_val = -1e6;
            float sum_exp = 0.0;

            // Find max (for stability)
            for (uint j = 0; j < op_c; j++)
            {
                max_val = max(max_val, State[state_base + op_a + j]);
            }

            // Compute exp and sum
            for (uint j = 0; j < op_c; j++)
            {
                float e = exp(State[state_base + op_a + j] - max_val);
                sum_exp += e;
                State[state_base + op_b + j] = e;
            }

            // Normalize
            for (uint j = 0; j < op_c; j++)
            {
                State[state_base + op_b + j] /= (sum_exp + 1e-6);
            }
            break;
        }

        case 3:  // OP_ADD
        {
            // Element-wise add: out = in1 + in2
            State[state_base + op_c] = 
                State[state_base + op_a] + State[state_base + op_b];
            break;
        }

        case 4:  // OP_MUL
        {
            // Element-wise mul: out = in1 * in2
            State[state_base + op_c] = 
                State[state_base + op_a] * State[state_base + op_b];
            break;
        }

        case 8:  // OP_HALT
        {
            // Stop execution
            return;
        }

        default:
            break;
        }

        // Synchronize threads (optional, depending on dependencies)
        GroupMemoryBarrierWithGroupSync();
    }
}
