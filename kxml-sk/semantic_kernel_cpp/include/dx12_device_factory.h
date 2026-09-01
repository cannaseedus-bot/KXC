// ============================================================================
// dx12_device_factory.h - DX12 Initialization & Hardware Fallback
// ============================================================================

#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <iostream>

using Microsoft::WRL::ComPtr;

namespace asx {

class DX12DeviceFactory {
public:
    struct DeviceContext {
        ComPtr<ID3D12Device> device;
        ComPtr<ID3D12CommandQueue> copy_queue;
        ComPtr<ID3D12CommandQueue> compute_queue;
        ComPtr<ID3D12CommandAllocator> compute_allocator;
        ComPtr<ID3D12GraphicsCommandList> compute_list;
        ComPtr<ID3D12Fence> fence;
        HANDLE fence_event;
        bool is_hardware = false;
    };

    static DeviceContext create_context();
};


} // namespace asx
