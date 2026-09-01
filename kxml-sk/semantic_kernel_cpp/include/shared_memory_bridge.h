// ============================================================================
// shared_memory_bridge.h - IPC Substrate (ASX v0.7 Phase 6)
// ============================================================================

#pragma once

#include <windows.h>
#include <string>
#include <iostream>
#include <vector>
#include <cstdint>

namespace asx {

// The header that sits at the beginning of the shared memory mapped file
struct SharedStateHeader {
    uint32_t version;
    uint32_t active_fold;
    uint32_t tick_count;
    float entropy;
    float attention;
    float pressure;
    float reserve[10]; // padding for future expansion
};

class SharedMemoryBridge {
private:
    HANDLE m_hMapFile;
    void* m_pBuf;
    std::string m_map_name;
    size_t m_size;

public:
    SharedMemoryBridge(const std::string& map_name, size_t size_bytes = 4096) 
        : m_map_name(map_name), m_size(size_bytes), m_hMapFile(NULL), m_pBuf(nullptr) {}

    ~SharedMemoryBridge() {
        if (m_pBuf) UnmapViewOfFile(m_pBuf);
        if (m_hMapFile) CloseHandle(m_hMapFile);
    }

    bool initialize_host() {
        m_hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE,    // use paging file
            NULL,                    // default security
            PAGE_READWRITE,          // read/write access
            0,                       // maximum object size (high-order DWORD)
            static_cast<DWORD>(m_size), // maximum object size (low-order DWORD)
            m_map_name.c_str());     // name of mapping object

        if (m_hMapFile == NULL) {
            if (GetLastError() == ERROR_ALREADY_EXISTS) {
                m_hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, m_map_name.c_str());
                if (m_hMapFile == NULL) {
                    std::cerr << "[WARN] SHM: Could not open existing file mapping object (" << GetLastError() << ").\n";
                    return false;
                }
            } else {
                std::cerr << "[WARN] SHM: Could not create file mapping object (" << GetLastError() << ").\n";
                return false;
            }
        }

        m_pBuf = MapViewOfFile(m_hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, m_size);


        if (m_pBuf == NULL) {
            std::cerr << "[WARN] SHM: Could not map view of file (" << GetLastError() << ").\n";
            CloseHandle(m_hMapFile);
            m_hMapFile = NULL;
            return false;
        }

        // Initialize header
        SharedStateHeader* header = static_cast<SharedStateHeader*>(m_pBuf);
        header->version = 1;
        header->active_fold = 0;
        header->tick_count = 0;
        header->entropy = 0.14f;
        header->attention = 0.72f;
        header->pressure = 0.34f;

        std::cout << "[INFO] SHM: Hosted Shared Memory Segment '" << m_map_name << "' (" << m_size << " bytes).\n";
        return true;
    }

    void update_state(uint32_t tick, uint32_t fold, float entropy, float attention, float pressure) {
        if (!m_pBuf) return;
        SharedStateHeader* header = static_cast<SharedStateHeader*>(m_pBuf);
        
        // In a real system, interlocked/atomic ops or a mutex should protect these writes
        header->tick_count = tick;
        header->active_fold = fold;
        header->entropy = entropy;
        header->attention = attention;
        header->pressure = pressure;
    }
};

} // namespace asx
