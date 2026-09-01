#pragma once
#include <cstdint>
#include <cstddef>

// IPC header and glyph entry definitions (binary-packed)
static const uint64_t GRAM_MAGIC = 0x47524D4D525F5447ULL; // "GRAMMR_TG" approx

#pragma pack(push,1)
struct IPCHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t totalSize;   // bytes of the mapping
    uint32_t glyphCount;  // number of glyph entries
    uint32_t status;      // 0=empty,1=ready,2=processed
    uint32_t reserved;
};

struct GlyphEntry {
    uint32_t codepoint;
    uint16_t glyphType;
    uint16_t pad;
    uint8_t  features[16];
    uint32_t reserved1;
    uint32_t reserved2;
};

#pragma pack(pop)

using GlyphOpcode = void(*)(GlyphEntry* input, GlyphEntry* output, void* memory);

