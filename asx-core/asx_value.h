#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace asx {

enum class ValueType {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
};

struct Value {
    ValueType type = ValueType::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<Value> array_value;
    std::map<std::string, Value> object_value;

    static Value make_null() { return {}; }
    static Value make_bool(bool value) {
        Value out;
        out.type = ValueType::Bool;
        out.bool_value = value;
        return out;
    }
    static Value make_number(double value) {
        Value out;
        out.type = ValueType::Number;
        out.number_value = value;
        return out;
    }
    static Value make_string(std::string value) {
        Value out;
        out.type = ValueType::String;
        out.string_value = std::move(value);
        return out;
    }
    static Value make_array() {
        Value out;
        out.type = ValueType::Array;
        return out;
    }
    static Value make_object() {
        Value out;
        out.type = ValueType::Object;
        return out;
    }

    bool is_object() const { return type == ValueType::Object; }
    bool is_array() const { return type == ValueType::Array; }
    bool is_string() const { return type == ValueType::String; }
};

inline const Value* get_child(const Value& value, const std::string& key) {
    if (!value.is_object()) return nullptr;
    auto it = value.object_value.find(key);
    return it == value.object_value.end() ? nullptr : &it->second;
}

inline const Value* get_path(const Value& value, const std::string& dotted_path) {
    const Value* current = &value;
    std::size_t offset = 0;
    while (offset < dotted_path.size()) {
        if (!current->is_object()) return nullptr;
        const std::string remaining = dotted_path.substr(offset);
        auto direct = current->object_value.find(remaining);
        if (direct != current->object_value.end()) return &direct->second;

        const Value* next = nullptr;
        std::size_t next_offset = std::string::npos;
        for (std::size_t dot = dotted_path.find('.', offset); dot != std::string::npos; dot = dotted_path.find('.', dot + 1)) {
            const std::string key = dotted_path.substr(offset, dot - offset);
            auto it = current->object_value.find(key);
            if (it != current->object_value.end()) {
                next = &it->second;
                next_offset = dot + 1;
            }
        }
        if (next == nullptr) return nullptr;
        current = next;
        offset = next_offset;
    }
    return current;
}

inline const std::string& require_string(const Value& value, const std::string& path) {
    const Value* child = get_path(value, path);
    if (child == nullptr || !child->is_string()) {
        throw std::runtime_error("missing string at path: " + path);
    }
    return child->string_value;
}

}  // namespace asx
