// ============================================================================
// dds_shard_loader.h - DX12 Tiled Weight Streaming
// ============================================================================

#pragma once

#include <d3d12.h>
#include <wrl.h>
#include <string>
#include <vector>
#include <map>

using Microsoft::WRL::ComPtr;

struct DdsShardMetadata {
    std::string filename;
    uint32_t width;
    uint32_t height;
    size_t size_bytes;
    DXGI_FORMAT format;
};

struct XShardMetadata {
    std::string filename;
    uint32_t shard_id;
    uint32_t weight_count;
    std::vector<float> weights;
};

class DdsShardLoader {
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_copy_queue;
    
    // Tiled Resource for the entire model (Virtual)
    ComPtr<ID3D12Resource> m_unified_weight_buffer;
    
    // CPU Fallback Buffer
    std::vector<float> m_cpu_weight_buffer;
    
    // Physical Heaps for "HOT" residency
    std::map<int, ComPtr<ID3D12Heap>> m_physical_heaps;
    
    // Mapping of folds to tiles
    std::map<int, DdsShardMetadata> m_shards;
    std::map<int, XShardMetadata> m_xshards;
    std::map<int, bool> m_fold_residency;
    
    const uint64_t TILE_SIZE = 64 * 1024; // DX12 Standard Tile Size: 64KB
    
public:
    DdsShardLoader(ID3D12Device* device, ID3D12CommandQueue* copyQueue) 
        : m_device(device), m_copy_queue(copyQueue) {}
        
    bool initialize_reserved_resource(size_t total_size_bytes) {
        // Law C: Create a Reserved Resource (Virtual Address Space)
        if (m_device) {
            D3D12_RESOURCE_DESC desc = {};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = total_size_bytes;
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE; // Required for tiling
            desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            
            HRESULT hr = m_device->CreateReservedResource(
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&m_unified_weight_buffer)
            );
            
            if (SUCCEEDED(hr)) return true;
        }

        // CPU Fallback
        m_cpu_weight_buffer.resize(total_size_bytes / sizeof(float), 0.0f);
        std::cout << "[INFO] Loader: Initialized CPU weight buffer (" << total_size_bytes / (1024*1024) << "MB).\n";
        return true;
    }
    
    void register_shard(int fold_id, const DdsShardMetadata& meta) {
        m_shards[fold_id] = meta;
        m_fold_residency[fold_id] = false;
    }

    void register_xshard(int fold_id, const XShardMetadata& meta) {
        m_xshards[fold_id] = meta;
        m_fold_residency[fold_id] = false;
    }
    
    bool load_xshard(int fold_id) {
        if (m_xshards.find(fold_id) == m_xshards.end()) return false;
        auto& meta = m_xshards[fold_id];
        
        std::ifstream ifs(meta.filename, std::ios::binary);
        if (!ifs) return false;
        
        uint32_t id, count;
        ifs.read(reinterpret_cast<char*>(&id), 4);
        ifs.read(reinterpret_cast<char*>(&count), 4);
        
        meta.shard_id = id;
        meta.weight_count = count;
        meta.weights.resize(count);
        ifs.read(reinterpret_cast<char*>(meta.weights.data()), count * sizeof(float));
        
        std::cout << "[INFO] Loader: Loaded XShard '" << meta.filename << "' (" << count << " weights).\n";
        return true;
    }

    
    bool update_fold_residency(int fold_id, bool hot) {
        if (m_shards.find(fold_id) == m_shards.end()) {
            std::cerr << "[WARN] DX12: Fold " << fold_id << " not registered.\n";
            return false;
        }

        if (hot == m_fold_residency[fold_id]) return true;
        
        if (!m_device || !m_copy_queue || !m_unified_weight_buffer) {
            std::cout << "[INFO] DX12: CPU fallback residency active for fold " << fold_id << ".\n";
            m_fold_residency[fold_id] = hot;
            return true;
        }

        uint32_t num_tiles = static_cast<uint32_t>((m_shards[fold_id].size_bytes + TILE_SIZE - 1) / TILE_SIZE);
        uint32_t start_tile = calculate_start_tile(fold_id);

        if (hot) {
            // ⟁ Law C: Promote to HOT
            D3D12_HEAP_DESC heap_desc = {};
            heap_desc.SizeInBytes = num_tiles * TILE_SIZE;
            heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            heap_desc.Flags = D3D12_HEAP_FLAG_NONE;

            ComPtr<ID3D12Heap> physical_heap;
            HRESULT hr = m_device->CreateHeap(&heap_desc, IID_PPV_ARGS(&physical_heap));
            if (FAILED(hr)) {
                std::cerr << "[WARN] DX12: Failed to create physical heap for fold " << fold_id << " (HRESULT: " << std::hex << hr << std::dec << ")\n";
                return false;
            }
            m_physical_heaps[fold_id] = physical_heap;
            
            D3D12_TILED_RESOURCE_COORDINATE start_coord = { start_tile, 0, 0, 0 };
            D3D12_TILE_REGION_SIZE region_size = { num_tiles, false, 0, 0, 0 };
            D3D12_TILE_RANGE_FLAGS range_flags = D3D12_TILE_RANGE_FLAG_NONE;
            uint32_t range_tile_count = num_tiles;
            
            m_copy_queue->UpdateTileMappings(
                m_unified_weight_buffer.Get(),
                1, &start_coord, &region_size,
                physical_heap.Get(),
                1, &range_flags, nullptr, &range_tile_count,
                D3D12_TILE_MAPPING_FLAG_NONE
            );
            
            std::cout << "[INFO] DX12: Fold " << fold_id << " mapped to HOT residency (" << num_tiles << " tiles).\n";
            m_fold_residency[fold_id] = true;
            
        } else {
            // ⟁ Law C: Demote to COLD (Null Mapping)
            D3D12_TILED_RESOURCE_COORDINATE start_coord = { start_tile, 0, 0, 0 };
            D3D12_TILE_REGION_SIZE region_size = { num_tiles, false, 0, 0, 0 };
            D3D12_TILE_RANGE_FLAGS range_flags = D3D12_TILE_RANGE_FLAG_NULL;
            uint32_t range_tile_count = num_tiles;

            m_copy_queue->UpdateTileMappings(
                m_unified_weight_buffer.Get(),
                1, &start_coord, &region_size,
                nullptr,
                1, &range_flags, nullptr, &range_tile_count,
                D3D12_TILE_MAPPING_FLAG_NONE
            );
            
            m_physical_heaps.erase(fold_id);
            m_fold_residency[fold_id] = false;
            std::cout << "[INFO] DX12: Fold " << fold_id << " unmapped (COLD).\n";
        }
        
        return true;
    }


    
private:
    uint32_t calculate_start_tile(int fold_id) {
        // Simple linear mapping of folds to the unified buffer
        uint32_t start = 0;
        for (int i = 0; i < fold_id; ++i) {
            if (m_shards.count(i)) {
                start += static_cast<uint32_t>((m_shards[i].size_bytes + TILE_SIZE - 1) / TILE_SIZE);
            }
        }
        return start;
    }
};

