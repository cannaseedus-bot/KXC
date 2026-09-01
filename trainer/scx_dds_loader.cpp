#include "scx_dds_loader.h"

#include "DDS.h"
#include "DDSTextureLoader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>
#include <limits>

namespace {

using DirectX::DDS_HEADER;
using DirectX::DDS_HEADER_DXT10;

constexpr uint32_t kDdsMagic = DirectX::DDS_MAGIC;
constexpr uint32_t kR32fFourCC = MAKEFOURCC('r', '3', '2', 'f');

void clear_metadata(SCX_DDS_METADATA* metadata) noexcept {
    if (!metadata) return;
    const auto size = std::min<size_t>(metadata->struct_size, sizeof(SCX_DDS_METADATA));
    if (size) std::memset(metadata, 0, size);
    metadata->struct_size = sizeof(SCX_DDS_METADATA);
}

HRESULT read_headers(const wchar_t* path, SCX_DDS_METADATA* metadata) noexcept {
    if (!path || !metadata) return E_INVALIDARG;

    std::ifstream input(path, std::ios::binary);
    if (!input) return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);

    uint32_t magic = 0;
    DDS_HEADER header{};
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || magic != kDdsMagic || header.size != sizeof(DDS_HEADER) ||
        header.ddspf.size != sizeof(DirectX::DDS_PIXELFORMAT)) {
        return HRESULT_FROM_WIN32(ERROR_INVALID_DATA);
    }

    metadata->width = header.width;
    metadata->height = header.height;
    metadata->depth = (header.flags & 0x00800000u) ? std::max(1u, header.depth) : 1u;
    metadata->mip_levels = std::max(1u, header.mipMapCount);
    metadata->array_size = 1;
    metadata->resource_dimension = D3D11_RESOURCE_DIMENSION_TEXTURE2D;
    metadata->dxgi_format = DXGI_FORMAT_UNKNOWN;
    metadata->data_offset = static_cast<uint32_t>(sizeof(uint32_t) + sizeof(DDS_HEADER));

    if (header.ddspf.flags & DDS_FOURCC && header.ddspf.fourCC == MAKEFOURCC('D','X','1','0')) {
        DDS_HEADER_DXT10 ext{};
        input.read(reinterpret_cast<char*>(&ext), sizeof(ext));
        if (!input || ext.arraySize == 0) return HRESULT_FROM_WIN32(ERROR_INVALID_DATA);
        metadata->dxgi_format = static_cast<uint32_t>(ext.dxgiFormat);
        metadata->resource_dimension = ext.resourceDimension;
        metadata->array_size = ext.arraySize;
        metadata->misc_flags = ext.miscFlag;
        metadata->is_cube = (ext.miscFlag & D3D11_RESOURCE_MISC_TEXTURECUBE) ? 1u : 0u;
        metadata->data_offset += static_cast<uint32_t>(sizeof(DDS_HEADER_DXT10));
    } else if (header.ddspf.flags & DDS_FOURCC && header.ddspf.fourCC == kR32fFourCC) {
        // SCXQDDS project container: legacy DDS header with raw R32_FLOAT data.
        metadata->dxgi_format = DXGI_FORMAT_R32_FLOAT;
    }
    return S_OK;
}

HRESULT load_scx_r32f(
    ID3D11Device* device,
    const wchar_t* path,
    const SCX_DDS_METADATA& metadata,
    ID3D11Resource** resource,
    ID3D11ShaderResourceView** view) noexcept {
    if (metadata.width == 0 || metadata.height == 0 || metadata.mip_levels != 1 ||
        metadata.array_size != 1 || metadata.depth != 1) {
        return HRESULT_FROM_WIN32(ERROR_INVALID_DATA);
    }

    const uint64_t row_bytes = static_cast<uint64_t>(metadata.width) * sizeof(float);
    const uint64_t payload_bytes = row_bytes * metadata.height;
    if (row_bytes > UINT_MAX || payload_bytes > SIZE_MAX) {
        return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);

    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    if (file_size < 0 || static_cast<uint64_t>(file_size) < payload_bytes) {
        return HRESULT_FROM_WIN32(ERROR_HANDLE_EOF);
    }
    uint64_t payload_offset = metadata.data_offset;
    // Some SCXQDDS writers record a 128-byte DDS boundary but emit a compact
    // 124-byte container prefix. Prefer the declared offset when valid; for a
    // short prefix, derive the only offset that can contain the full payload.
    if (payload_offset > static_cast<uint64_t>(file_size) ||
        static_cast<uint64_t>(file_size) - payload_offset < payload_bytes) {
        payload_offset = static_cast<uint64_t>(file_size) - payload_bytes;
    }
    input.seekg(static_cast<std::streamoff>(payload_offset), std::ios::beg);
    std::vector<uint8_t> payload(static_cast<size_t>(payload_bytes));
    input.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (input.gcount() != static_cast<std::streamsize>(payload.size())) {
        return HRESULT_FROM_WIN32(ERROR_HANDLE_EOF);
    }

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = metadata.width;
    desc.Height = metadata.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = payload.data();
    init.SysMemPitch = static_cast<UINT>(row_bytes);
    init.SysMemSlicePitch = static_cast<UINT>(payload_bytes);

    ID3D11Texture2D* texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&desc, &init, &texture);
    if (FAILED(hr)) return hr;

    ID3D11ShaderResourceView* texture_view = nullptr;
    hr = device->CreateShaderResourceView(texture, nullptr, &texture_view);
    if (FAILED(hr)) {
        texture->Release();
        return hr;
    }
    *resource = texture;
    *view = texture_view;
    return S_OK;
}

} // namespace

extern "C" SCX_DDS_API HRESULT __cdecl scx_dds_inspect_file(
    const wchar_t* path, SCX_DDS_METADATA* metadata) noexcept {
    if (!metadata) return E_INVALIDARG;
    clear_metadata(metadata);
    return read_headers(path, metadata);
}

extern "C" SCX_DDS_API HRESULT __cdecl scx_dds_load_texture(
    ID3D11Device* device,
    const wchar_t* path,
    ID3D11Resource** resource,
    ID3D11ShaderResourceView** view,
    SCX_DDS_METADATA* metadata) noexcept {
    if (!device || !path || !resource || !view) return E_INVALIDARG;
    *resource = nullptr;
    *view = nullptr;
    if (metadata) {
        clear_metadata(metadata);
        const HRESULT inspect_hr = read_headers(path, metadata);
        if (FAILED(inspect_hr)) return inspect_hr;
    }
    SCX_DDS_METADATA local_metadata{};
    local_metadata.struct_size = sizeof(local_metadata);
    if (!metadata) {
        const HRESULT inspect_hr = read_headers(path, &local_metadata);
        if (FAILED(inspect_hr)) return inspect_hr;
        metadata = &local_metadata;
    }
    if (metadata->dxgi_format == DXGI_FORMAT_R32_FLOAT) {
        return load_scx_r32f(device, path, *metadata, resource, view);
    }
    return DirectX::CreateDDSTextureFromFileEx(
        device, path, 0, D3D11_USAGE_DEFAULT,
        D3D11_BIND_SHADER_RESOURCE, 0, 0,
        DirectX::DDS_LOADER_DEFAULT, resource, view, nullptr);
}

extern "C" SCX_DDS_API const char* __cdecl scx_dds_loader_version() noexcept {
    return "scx-dds-loader/1; directxtk-d3d11; metadata+resident-texture";
}
