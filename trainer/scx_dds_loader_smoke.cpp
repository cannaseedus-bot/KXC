#include "scx_dds_loader.h"
#include "DDS.h"

#include <d3d11.h>
#include <cstdio>
#include <cwchar>
#include <fstream>
#include <vector>

static bool write_fixture(const wchar_t* path) {
    DirectX::DDS_HEADER header{};
    header.size = sizeof(header);
    header.flags = DDS_HEADER_FLAGS_TEXTURE | DDS_HEADER_FLAGS_PITCH;
    header.height = 2;
    header.width = 2;
    header.pitchOrLinearSize = 2u * 4u * sizeof(float);
    header.mipMapCount = 1;
    header.ddspf = DirectX::DDSPF_DX10;
    header.caps = DDS_SURFACE_FLAGS_TEXTURE;

    DirectX::DDS_HEADER_DXT10 ext{};
    ext.dxgiFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
    ext.resourceDimension = D3D11_RESOURCE_DIMENSION_TEXTURE2D;
    ext.arraySize = 1;

    const float pixels[16] = {
        1.f, 0.f, 0.f, 1.f,  0.f, 1.f, 0.f, 1.f,
        0.f, 0.f, 1.f, 1.f,  1.f, 1.f, 1.f, 1.f
    };
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    const uint32_t magic = DirectX::DDS_MAGIC;
    output.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(&ext), sizeof(ext));
    output.write(reinterpret_cast<const char*>(pixels), sizeof(pixels));
    return static_cast<bool>(output);
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2 && argc != 3) {
        ::fwprintf(stderr, L"usage: scx_dds_loader_smoke.exe <file.dds> | --generate <file.dds>\n");
        return 2;
    }

    if (argc == 3 && wcscmp(argv[1], L"--generate") == 0) {
        if (!write_fixture(argv[2])) return 3;
        argv[1] = argv[2];
    } else if (argc != 2) {
        return 2;
    }

    SCX_DDS_METADATA metadata{};
    metadata.struct_size = sizeof(metadata);
    HRESULT hr = scx_dds_inspect_file(argv[1], &metadata);
    if (FAILED(hr)) {
        ::fwprintf(stderr, L"inspect failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return 3;
    }

    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    D3D_FEATURE_LEVEL level{};
    const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0
    };
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        0, levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
        &device, &level, &context);
    if (FAILED(hr)) {
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr,
            0, levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &device, &level, &context);
    }
    if (FAILED(hr)) {
        ::fwprintf(stderr, L"D3D11 device failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return 4;
    }

    ID3D11Resource* resource = nullptr;
    ID3D11ShaderResourceView* view = nullptr;
    hr = scx_dds_load_texture(device, argv[1], &resource, &view, &metadata);
    if (FAILED(hr)) {
        ::fwprintf(stderr, L"load failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        context->Release();
        device->Release();
        return 5;
    }

    D3D11_RESOURCE_DIMENSION actual_dimension{};
    resource->GetType(&actual_dimension);
    std::printf("ok version=%s width=%u height=%u depth=%u mips=%u array=%u dxgi=%u dimension=%u actual=%u feature_level=%x\n",
        scx_dds_loader_version(), metadata.width, metadata.height, metadata.depth,
        metadata.mip_levels, metadata.array_size, metadata.dxgi_format,
        metadata.resource_dimension, actual_dimension, static_cast<unsigned>(level));

    view->Release();
    resource->Release();
    context->Release();
    device->Release();
    return 0;
}
