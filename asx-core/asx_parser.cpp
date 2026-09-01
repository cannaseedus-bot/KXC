#include "asx_parser.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace asx {

namespace {

struct Line {
    int indent = 0;
    std::string text;
};

std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) ++start;
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return value.substr(start, end - start);
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.rfind(prefix, 0) == 0;
}

bool is_array_item(const std::string& text) {
    return text == "-" || starts_with(text, "- ");
}

Value parse_scalar(const std::string& token) {
    if (token == "true") return Value::make_bool(true);
    if (token == "false") return Value::make_bool(false);
    if (token == "null") return Value::make_null();
    if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
        std::string out;
        out.reserve(token.size() - 2);
        for (std::size_t i = 1; i + 1 < token.size(); ++i) {
            char c = token[i];
            if (c == '\\' && i + 1 < token.size() - 1) {
                char n = token[++i];
                switch (n) {
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case '\\': out.push_back('\\'); break;
                    case '"': out.push_back('"'); break;
                    default: out.push_back(n); break;
                }
            } else {
                out.push_back(c);
            }
        }
        return Value::make_string(out);
    }

    char* end = nullptr;
    const double number = std::strtod(token.c_str(), &end);
    if (end != nullptr && *end == '\0' && !token.empty()) {
        return Value::make_number(number);
    }
    return Value::make_string(token);
}

std::vector<std::string> split_inline_array(const std::string& raw) {
    std::vector<std::string> items;
    std::string current;
    bool in_string = false;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        char c = raw[i];
        if (c == '"' && (i == 0 || raw[i - 1] != '\\')) in_string = !in_string;
        if (c == ',' && !in_string) {
            items.push_back(trim(current));
            current.clear();
            continue;
        }
        current.push_back(c);
    }
    if (!trim(current).empty()) items.push_back(trim(current));
    return items;
}

Value parse_inline_value(const std::string& token) {
    const std::string value = trim(token);
    if (value.size() >= 2 && value.front() == '[' && value.back() == ']') {
        Value array = Value::make_array();
        for (const std::string& item : split_inline_array(value.substr(1, value.size() - 2))) {
            array.array_value.push_back(parse_scalar(item));
        }
        return array;
    }
    return parse_scalar(value);
}

class Parser {
  public:
    explicit Parser(std::vector<Line> lines) : lines_(std::move(lines)) {}

    Value parse() {
        Value root = Value::make_object();
        while (index_ < lines_.size()) {
            const Line& line = lines_[index_];
            if (!starts_with(line.text, "@@")) {
                throw std::runtime_error("top-level line must begin with @@: " + line.text);
            }
            const std::string key = line.text.substr(2);
            ++index_;
            Value block = Value::make_object();
            if (index_ < lines_.size() && lines_[index_].indent > line.indent) {
                block = parse_block(lines_[index_].indent);
            }
            root.object_value.insert({key, block});
        }
        return root;
    }

  private:
    Value parse_block(int indent) {
        Value object = Value::make_object();
        while (index_ < lines_.size()) {
            const Line& line = lines_[index_];
            if (line.indent < indent) break;
            if (line.indent > indent) {
                throw std::runtime_error("unexpected indentation at line: " + line.text);
            }
            if (is_array_item(line.text)) {
                throw std::runtime_error("array item cannot appear without owning key");
            }

            const std::size_t colon = line.text.find(':');
            if (colon != std::string::npos) {
                const std::string key = trim(line.text.substr(0, colon));
                const std::string rest = trim(line.text.substr(colon + 1));
                ++index_;
                if (!rest.empty()) {
                    object.object_value[key] = parse_inline_value(rest);
                    continue;
                }
                if (index_ < lines_.size() && lines_[index_].indent > indent) {
                    if (is_array_item(lines_[index_].text)) {
                        object.object_value[key] = parse_array(lines_[index_].indent);
                    } else {
                        object.object_value[key] = parse_block(lines_[index_].indent);
                    }
                } else {
                    object.object_value[key] = Value::make_object();
                }
                continue;
            }

            const std::string key = trim(line.text);
            ++index_;
            if (index_ < lines_.size() && lines_[index_].indent > indent) {
                if (is_array_item(lines_[index_].text)) {
                    object.object_value[key] = parse_array(lines_[index_].indent);
                } else {
                    object.object_value[key] = parse_block(lines_[index_].indent);
                }
            } else {
                object.object_value[key] = Value::make_object();
            }
        }
        return object;
    }

    Value parse_array(int indent) {
        Value array = Value::make_array();
        while (index_ < lines_.size()) {
            const Line& line = lines_[index_];
            if (line.indent < indent) break;
            if (line.indent > indent || !is_array_item(line.text)) {
                throw std::runtime_error("malformed array item: " + line.text);
            }
            const std::string item = line.text == "-" ? "" : trim(line.text.substr(2));
            ++index_;
            if (!item.empty()) {
                array.array_value.push_back(parse_inline_value(item));
                continue;
            }
            if (index_ < lines_.size() && lines_[index_].indent > indent) {
                if (is_array_item(lines_[index_].text)) {
                    array.array_value.push_back(parse_array(lines_[index_].indent));
                } else {
                    array.array_value.push_back(parse_block(lines_[index_].indent));
                }
            } else {
                array.array_value.push_back(Value::make_null());
            }
        }
        return array;
    }

    std::vector<Line> lines_;
    std::size_t index_ = 0;
};

std::vector<Line> lex_lines(const std::string& source) {
    std::vector<Line> lines;
    std::istringstream input(source);
    std::string raw;
    while (std::getline(input, raw)) {
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        int indent = 0;
        while (indent < static_cast<int>(raw.size()) && raw[indent] == ' ') ++indent;
        const std::string text = trim(raw.substr(indent));
        if (text.empty() || starts_with(text, "#")) continue;
        lines.push_back({indent, text});
    }
    return lines;
}

std::string read_file_text(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to read file: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

Value parse_document(const std::string& source) {
    Parser parser(lex_lines(source));
    return parser.parse();
}

Value parse_file(const std::string& path) {
    return parse_document(read_file_text(path));
}

}  // namespace asx
