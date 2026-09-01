#pragma once

#include <Windows.h>
#include <d3d11.h>
#include <cstdint>

#ifdef SCX_DDS_LOADER_BUILD
#define SCX_DDS_API __declspec(dllexport)
#else
#define SCX_DDS_API __declspec(dllimport)
#endif

extern "C" {

// ABI-stable physical DDS metadata. Semantic identity belongs in the caller's
// field graph; this structure reports only the resident resource payload.
struct SCX_DDS_METADATA {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_levels;
    uint32_t array_size;
    uint32_t dxgi_format;
    uint32_t resource_dimension;
    uint32_t misc_flags;
    uint32_t data_offset;
    uint32_t is_cube;
    uint32_t reserved[5];
};

SCX_DDS_API HRESULT __cdecl scx_dds_inspect_file(
    const wchar_t* path,
    SCX_DDS_METADATA* metadata) noexcept;

SCX_DDS_API HRESULT __cdecl scx_dds_load_texture(
    ID3D11Device* device,
    const wchar_t* path,
    ID3D11Resource** resource,
    ID3D11ShaderResourceView** view,
    SCX_DDS_METADATA* metadata) noexcept;

SCX_DDS_API const char* __cdecl scx_dds_loader_version() noexcept;

}
