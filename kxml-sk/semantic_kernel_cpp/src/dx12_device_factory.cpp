// ============================================================================
// dx12_device_factory.cpp - DX12 Hardware Logic (ASX v0.7)
// ============================================================================

#include "../include/dx12_device_factory.h"

namespace asx {

DX12DeviceFactory::DeviceContext DX12DeviceFactory::create_context() {
    DeviceContext ctx;
    
    #ifdef WIN32
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        std::cerr << "[WARN] DX12: Failed to create DXGI Factory.\n";
        return ctx;
    }

    ComPtr<IDXGIAdapter1> adapter;
    for (uint32_t i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ctx.device)))) {
            std::cout << "[INFO] DX12: Hardware Device Initialized.\n";
            ctx.is_hardware = true;
            break;
        }
    }

    if (ctx.is_hardware) {
        // 1. Create Copy Queue (for tiled weight residency)
        D3D12_COMMAND_QUEUE_DESC copy_desc = {};
        copy_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        copy_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        ctx.device->CreateCommandQueue(&copy_desc, IID_PPV_ARGS(&ctx.copy_queue));

        // 2. Create Compute Queue (for inference/fusion)
        D3D12_COMMAND_QUEUE_DESC compute_desc = {};
        compute_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        compute_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        ctx.device->CreateCommandQueue(&compute_desc, IID_PPV_ARGS(&ctx.compute_queue));

        // 3. Create Command Allocator and List for Compute
        ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&ctx.compute_allocator));
        ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, ctx.compute_allocator.Get(), nullptr, IID_PPV_ARGS(&ctx.compute_list));
        ctx.compute_list->Close(); // Start in closed state

        // 4. Create Synchronization Objects (Fence)
        ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx.fence));
        ctx.fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        std::cout << "[INFO] DX12: Command Queues and Synchronization ready.\n";
    } else {
        std::cerr << "[WARN] DX12: No hardware device found. Falling back to CPU simulation.\n";
    }
    #else
    std::cerr << "[WARN] DX12: Platform not supported. Falling back to CPU simulation.\n";
    #endif

    return ctx;
}

} // namespace asx
