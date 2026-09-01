#include "asx_canonical.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace asx {

namespace {

std::string escape_json(const std::string& input) {
    std::ostringstream out;
    for (char c : input) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

std::string serialize_number(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite number in canonical serialization");
    std::ostringstream out;
    out << std::setprecision(15) << value;
    std::string text = out.str();
    if (text.find('.') != std::string::npos) {
        while (!text.empty() && text.back() == '0') text.pop_back();
        if (!text.empty() && text.back() == '.') text.pop_back();
    }
    if (text.empty()) text = "0";
    return text;
}

}  // namespace

std::string canonical_json(const Value& value) {
    switch (value.type) {
        case ValueType::Null:
            return "null";
        case ValueType::Bool:
            return value.bool_value ? "true" : "false";
        case ValueType::Number:
            return serialize_number(value.number_value);
        case ValueType::String:
            return "\"" + escape_json(value.string_value) + "\"";
        case ValueType::Array: {
            std::ostringstream out;
            out << "[";
            for (std::size_t i = 0; i < value.array_value.size(); ++i) {
                if (i > 0) out << ",";
                out << canonical_json(value.array_value[i]);
            }
            out << "]";
            return out.str();
        }
        case ValueType::Object: {
            std::ostringstream out;
            out << "{";
            bool first = true;
            for (const auto& entry : value.object_value) {
                if (!first) out << ",";
                first = false;
                out << "\"" << escape_json(entry.first) << "\":" << canonical_json(entry.second);
            }
            out << "}";
            return out.str();
        }
    }
    throw std::runtime_error("unsupported value type");
}

}  // namespace asx
