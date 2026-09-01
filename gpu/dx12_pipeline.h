#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <vector>
#include <unordered_map>

using Microsoft::WRL::ComPtr;

namespace asx {
namespace gpu {

/**
 * @brief DirectX 12 Pipeline for Fabric execution
 * 
 * Manages:
 * - Device and command queues (compute + copy)
 * - Descriptor heaps (CBV/SRV/UAV)
 * - Root signatures and pipeline state objects
 * - Command list recording and submission
 * - Async compute execution with synchronization
 * 
 * Execution model:
 *   Copy Queue:    Upload buffers (async, doesn't block GPU)
 *   Compute Queue: Run compute shaders (parallel with copy)
 *   
 * Dependencies:
 *   - Windows SDK 10.0 or higher
 *   - D3D12 driver with async compute support
 */

/**
 * @brief GPU buffer wrapper (device memory)
 */
struct GPUBuffer
{
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Resource> upload_heap;  // For CPU → GPU transfers
    
    uint32_t descriptor_index = UINT32_MAX;
    size_t size_bytes = 0;
    
    bool is_uav = false;  // Unordered access view (writable)
};

/**
 * @brief Compiled shader program (all entry points)
 */
struct ShaderProgram
{
    ComPtr<ID3DBlob> bytecode;
    std::string name;
    size_t size_bytes = 0;
};

/**
 * @brief Descriptor heap wrapper with allocation tracking
 */
class DescriptorHeapManager
{
public:
    /**
     * @brief Initialize descriptor heap
     * 
     * @param device D3D12 device
     * @param heap_size Number of descriptors
     * @return true if successful
     */
    bool Initialize(ID3D12Device* device, uint32_t heap_size);

    /**
     * @brief Allocate descriptor and return index
     * 
     * @return Descriptor index (for use in descriptor tables)
     */
    uint32_t AllocateDescriptor();

    /**
     * @brief Get CPU-side descriptor handle
     * 
     * @param index Descriptor index
     * @return CPU handle
     */
    D3D12_CPU_DESCRIPTOR_HANDLE GetCPUHandle(uint32_t index);

    /**
     * @brief Get GPU-side descriptor handle
     * 
     * @param index Descriptor index
     * @return GPU handle
     */
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle(uint32_t index);

    /**
     * @brief Get heap itself
     * 
     * @return ID3D12DescriptorHeap
     */
    ID3D12DescriptorHeap* GetHeap() { return heap_.Get(); }

    /**
     * @brief Reset allocator (for new frame)
     */
    void Reset();

private:
    ComPtr<ID3D12DescriptorHeap> heap_;
    uint32_t heap_size_ = 0;
    uint32_t current_index_ = 0;
    uint32_t descriptor_size_ = 0;
};

/**
 * @brief Async compute command list wrapper
 */
class ComputeCommandList
{
public:
    /**
     * @brief Record command list
     * 
     * @param device D3D12 device
     * @param allocator Command allocator
     * @return true if successful
     */
    bool Initialize(ID3D12Device* device, ID3D12CommandAllocator* allocator);

    /**
     * @brief Set compute root signature
     * 
     * @param root_sig Root signature
     */
    void SetRootSignature(ID3D12RootSignature* root_sig);

    /**
     * @brief Set compute shader (PSO)
     * 
     * @param pso Pipeline state object
     */
    void SetPipelineState(ID3D12PipelineState* pso);

    /**
     * @brief Set descriptor heap
     * 
     * @param heap Descriptor heap
     */
    void SetDescriptorHeap(ID3D12DescriptorHeap* heap);

    /**
     * @brief Set root parameter (constant buffer)
     * 
     * @param param_idx Parameter index in root signature
     * @param gpu_handle GPU descriptor handle
     */
    void SetRootDescriptorTable(uint32_t param_idx, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle);

    /**
     * @brief Dispatch compute shader
     * 
     * @param x Thread groups in X dimension
     * @param y Thread groups in Y dimension
     * @param z Thread groups in Z dimension
     */
    void Dispatch(uint32_t x, uint32_t y, uint32_t z);

    /**
     * @brief Finish recording and submit
     * 
     * @param queue Command queue to submit to
     * @param fence Fence for synchronization
     * @param fence_value Next fence value
     * @return Fence value after submission
     */
    uint64_t Finish(ID3D12CommandQueue* queue, ID3D12Fence* fence, uint64_t fence_value);

    /**
     * @brief Get command list
     * 
     * @return ID3D12GraphicsCommandList
     */
    ID3D12GraphicsCommandList* Get() { return cmdlist_.Get(); }

private:
    ComPtr<ID3D12GraphicsCommandList> cmdlist_;
    ComPtr<ID3D12CommandAllocator> allocator_;
};

/**
 * @brief DirectX 12 GPU Pipeline Manager
 * 
 * Central hub for GPU execution: device, queues, command lists, synchronization
 */
class DX12Pipeline
{
public:
    /**
     * @brief Initialize D3D12 pipeline
     * 
     * @param enable_debug Enable debug layer and GPU-based validation
     * @return true if successful
     */
    static bool InitializeDevice(bool enable_debug = false);

    /**
     * @brief Get singleton instance
     * 
     * @return DX12Pipeline instance
     */
    static DX12Pipeline& Get();

    /**
     * @brief Allocate GPU buffer
     * 
     * @param size_bytes Buffer size
     * @param is_uav Whether buffer is writable (UAV)
     * @param initial_data Optional CPU data to upload
     * @return GPU buffer
     */
    GPUBuffer AllocateBuffer(
        size_t size_bytes,
        bool is_uav = false,
        const void* initial_data = nullptr);

    /**
     * @brief Upload data to GPU buffer
     * 
     * @param buffer Target buffer
     * @param data Source data
     * @param size Size to upload
     * @param offset Offset in buffer
     */
    void UploadBuffer(const GPUBuffer& buffer, const void* data, size_t size, size_t offset = 0);

    /**
     * @brief Download data from GPU buffer
     * 
     * @param buffer Source buffer
     * @param data Destination data
     * @param size Size to download
     * @param offset Offset in buffer
     */
    void DownloadBuffer(const GPUBuffer& buffer, void* data, size_t size, size_t offset = 0);

    /**
     * @brief Create root signature from descriptor layout
     * 
     * @param descriptors Descriptor parameter layout
     * @return Root signature
     */
    ComPtr<ID3D12RootSignature> CreateRootSignature(
        const std::vector<D3D12_ROOT_PARAMETER>& descriptors);

    /**
     * @brief Compile shader from HLSL code
     * 
     * @param hlsl_code HLSL source code
     * @param entry_point Entry point function name
     * @param target Shader target (e.g., "cs_5_1")
     * @return Compiled shader bytecode
     */
    ComPtr<ID3DBlob> CompileShader(
        const std::string& hlsl_code,
        const std::string& entry_point,
        const std::string& target);

    /**
     * @brief Create compute PSO
     * 
     * @param root_sig Root signature
     * @param shader_bytecode Compiled shader
     * @return Pipeline state object
     */
    ComPtr<ID3D12PipelineState> CreateComputePSO(
        ID3D12RootSignature* root_sig,
        ID3DBlob* shader_bytecode);

    /**
     * @brief Create SRV (Structured Buffer)
     * 
     * @param buffer GPU buffer
     * @param descriptor_index Descriptor heap index
     * @param element_count Number of elements
     * @param element_size Size per element (bytes)
     */
    void CreateSRV(
        const GPUBuffer& buffer,
        uint32_t descriptor_index,
        uint32_t element_count,
        uint32_t element_size);

    /**
     * @brief Create UAV (RW Structured Buffer)
     * 
     * @param buffer GPU buffer
     * @param descriptor_index Descriptor heap index
     * @param element_count Number of elements
     * @param element_size Size per element
     */
    void CreateUAV(
        const GPUBuffer& buffer,
        uint32_t descriptor_index,
        uint32_t element_count,
        uint32_t element_size);

    /**
     * @brief Create CBV (Constant Buffer)
     * 
     * @param buffer GPU buffer
     * @param descriptor_index Descriptor heap index
     * @param size_bytes Buffer size
     */
    void CreateCBV(
        const GPUBuffer& buffer,
        uint32_t descriptor_index,
        uint32_t size_bytes);

    /**
     * @brief Execute compute shader
     * 
     * @param pso Pipeline state object
     * @param root_sig Root signature
     * @param descriptor_table Descriptor table GPU handle
     * @param thread_groups_x Thread groups in X
     * @param thread_groups_y Thread groups in Y (default 1)
     * @param thread_groups_z Thread groups in Z (default 1)
     * @return Fence value for synchronization
     */
    uint64_t ExecuteCompute(
        ID3D12PipelineState* pso,
        ID3D12RootSignature* root_sig,
        D3D12_GPU_DESCRIPTOR_HANDLE descriptor_table,
        uint32_t thread_groups_x,
        uint32_t thread_groups_y = 1,
        uint32_t thread_groups_z = 1);

    /**
     * @brief Wait for GPU work to complete
     * 
     * @param fence_value Fence value to wait for
     * @param timeout_ms Timeout in milliseconds (0 = infinite)
     * @return true if waited successfully, false if timeout
     */
    bool WaitForGPU(uint64_t fence_value, uint32_t timeout_ms = 0);

    /**
     * @brief Flush all pending GPU commands
     */
    void FlushGPU();

    /**
     * @brief Get device
     * 
     * @return ID3D12Device
     */
    ID3D12Device* GetDevice() { return device_.Get(); }

    /**
     * @brief Shutdown pipeline (cleanup)
     */
    void Shutdown();

private:
    DX12Pipeline() = default;
    ~DX12Pipeline();

    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> compute_queue_;
    ComPtr<ID3D12CommandQueue> copy_queue_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_ = nullptr;
    uint64_t fence_value_ = 0;

    std::unique_ptr<DescriptorHeapManager> descriptor_manager_;
    std::unique_ptr<ComputeCommandList> compute_cmdlist_;

    // Shader cache
    std::unordered_map<std::string, ComPtr<ID3DBlob>> shader_cache_;

    bool CreateQueues();
    bool CreateFence();
};

}  // namespace gpu
}  // namespace asx

#endif  // ASX_GPU_DX12_PIPELINE_H
