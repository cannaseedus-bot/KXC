#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "ir.h"
#include "executor.h"

// Simple symbol table that reserves indices into Buffers.i64
struct SymbolTable {
    std::unordered_map<std::string, uint32_t> map_i64;
    std::unordered_map<std::string, uint32_t> map_f32;
    uint32_t next_i64 = 0;
    uint32_t next_f32 = 0;
    Buffers* buf = nullptr;
    SymbolTable() = default;
    SymbolTable(Buffers* b):next_i64(0), next_f32(0), buf(b) {}

    uint32_t resolve_i64(const std::string &name){
        auto it = map_i64.find(name);
        if (it != map_i64.end()) return it->second;
        uint32_t id = next_i64++;
        map_i64[name] = id;
        if (buf) {
            if (buf->i64.size() < next_i64) buf->i64.resize(next_i64);
        }
        return id;
    }

    uint32_t resolve_f32(const std::string &name){
        auto it = map_f32.find(name);
        if (it != map_f32.end()) return it->second;
        uint32_t id = next_f32++;
        map_f32[name] = id;
        if (buf) {
            if (buf->f32.size() < next_f32) buf->f32.resize(next_f32);
        }
        return id;
    }
};

struct CompiledProgram {
    std::vector<IRNode> nodes;
    Buffers buffers;
    SymbolTable symbols;

    CompiledProgram():symbols(&buffers){}
};

// Compile high-level ExecutionNode tree into numeric IRNodes (K'UHUL IR)
// - validates simple opcode set
// - lowers literals into buffers and symbolic names into symbol table
// returns true on success
bool compileProgram(ExecutionNode &root, CompiledProgram &out);
