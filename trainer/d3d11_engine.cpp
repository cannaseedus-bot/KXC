#include "d3d11_engine.h"
#include <dxgi.h>
#include <iostream>
#include <string>

#pragma comment(lib, "d3d11")
#pragma comment(lib, "dxgi")

bool D3D11Engine::init(bool forceWarp, bool verboseLog) {
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_DRIVER_TYPE dtype = forceWarp ? D3D_DRIVER_TYPE_WARP : D3D_DRIVER_TYPE_HARDWARE;

    // Grab adapter name via DXGI before creating the device
    if (!forceWarp) {
        ComPtr<IDXGIFactory> factory;
        if (SUCCEEDED(CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(factory.GetAddressOf())))) {
            ComPtr<IDXGIAdapter> adapter;
            if (SUCCEEDED(factory->EnumAdapters(0, adapter.GetAddressOf()))) {
                DXGI_ADAPTER_DESC desc{};
                if (SUCCEEDED(adapter->GetDesc(&desc))) {
                    int len = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                                                  nullptr, 0, nullptr, nullptr);
                    if (len > 1) {
                        adapterName_.resize(static_cast<size_t>(len - 1));
                        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                                            adapterName_.data(), len, nullptr, nullptr);
                    }
                }
            }
        }
    } else {
        adapterName_ = "WARP";
    }

    HRESULT hr = D3D11CreateDevice(
        nullptr, dtype, nullptr,
        D3D11_CREATE_DEVICE_SINGLETHREADED,
        levels, static_cast<UINT>(_countof(levels)),
        D3D11_SDK_VERSION,
        device_.GetAddressOf(), &featureLevel_, ctx_.GetAddressOf()
    );

    // Fall back to WARP if hardware device creation fails
    if (FAILED(hr)) {
        hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr,
            D3D11_CREATE_DEVICE_SINGLETHREADED,
            levels, static_cast<UINT>(_countof(levels)),
            D3D11_SDK_VERSION,
            device_.GetAddressOf(), &featureLevel_, ctx_.GetAddressOf()
        );
        if (SUCCEEDED(hr)) {
            usedWarp_ = true;
            adapterName_ = "WARP (hw fallback)";
        }
    }

    if (FAILED(hr)) {
        char buf[64];
        snprintf(buf, sizeof(buf), "D3D11CreateDevice HRESULT 0x%08X", static_cast<unsigned>(hr));
        initReason_ = buf;
        return false;
    }

    if (verboseLog)
        std::cerr << "[D3D11Engine] " << adapterName_
                  << "  FL=0x" << std::hex << featureLevel_ << std::dec
                  << (usedWarp_ ? "  (WARP)" : "") << "\n";
    return true;
}
