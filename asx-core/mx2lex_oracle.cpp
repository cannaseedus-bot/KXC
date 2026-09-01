#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "asx_canonical.h"
#include "asx_parser.h"
#include "asx_value.h"
#include "sha256.h"

namespace {

struct Transition {
    std::string from;
    std::string sym;
    std::string to;
};

struct DfaDocument {
    std::string id;
    std::string start;
    std::set<std::string> accept;
    std::vector<Transition> transitions;
    std::string document_hash;
};

struct OracleRequest {
    std::string id;
    std::vector<std::string> tokens;
    std::string document_hash;
};

std::string escape_json(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

std::vector<std::string> require_string_array(const asx::Value& value, const std::string& path) {
    const asx::Value* node = asx::get_path(value, path);
    if (node == nullptr || !node->is_array()) throw std::runtime_error("expected string array at path: " + path);
    std::vector<std::string> out;
    out.reserve(node->array_value.size());
    for (const auto& item : node->array_value) {
        if (!item.is_string()) throw std::runtime_error("expected string array entry at path: " + path);
        out.push_back(item.string_value);
    }
    return out;
}

DfaDocument load_dfa(const std::string& path) {
    const asx::Value document = asx::parse_file(path);
    const asx::Value* root = asx::get_child(document, "mx2lex.dfa");
    if (root == nullptr || !root->is_object()) throw std::runtime_error("missing @@mx2lex.dfa block");

    DfaDocument out;
    out.id = asx::require_string(*root, "@id");
    out.start = asx::require_string(*root, "start");
    out.document_hash = asx::sha256_hex(asx::canonical_json(document));
    for (const auto& state : require_string_array(*root, "accept")) out.accept.insert(state);

    const asx::Value* transitions = asx::get_path(*root, "transitions");
    if (transitions == nullptr || !transitions->is_array()) throw std::runtime_error("expected transitions array");
    for (const auto& item : transitions->array_value) {
        if (!item.is_object()) throw std::runtime_error("transition entry must be object");
        out.transitions.push_back({
            asx::require_string(item, "from"),
            asx::require_string(item, "sym"),
            asx::require_string(item, "to"),
        });
    }
    return out;
}

OracleRequest load_request(const std::string& path) {
    const asx::Value document = asx::parse_file(path);
    const asx::Value* root = asx::get_child(document, "mx2lex.oracle.request");
    if (root == nullptr || !root->is_object()) throw std::runtime_error("missing @@mx2lex.oracle.request block");

    OracleRequest out;
    out.id = asx::require_string(*root, "@id");
    out.tokens = require_string_array(*root, "tokens");
    out.document_hash = asx::sha256_hex(asx::canonical_json(document));
    return out;
}

struct OracleResult {
    bool pass = false;
    std::string final_state;
    std::vector<std::string> trace;
    std::string trace_hash;
};

OracleResult run_oracle(const DfaDocument& dfa, const OracleRequest& request) {
    std::map<std::string, std::map<std::string, std::string>> transitions;
    for (const auto& transition : dfa.transitions) {
        transitions[transition.from][transition.sym] = transition.to;
    }

    std::string state = dfa.start;
    OracleResult result;
    result.trace.push_back(state);

    for (const auto& token : request.tokens) {
        const auto state_it = transitions.find(state);
        if (state_it == transitions.end()) {
            result.final_state = state;
            result.pass = false;
            asx::Value trace_doc = asx::Value::make_object();
            trace_doc.object_value["@state_trace"] = asx::Value::make_array();
            for (const auto& entry : result.trace) trace_doc.object_value["@state_trace"].array_value.push_back(asx::Value::make_string(entry));
            result.trace_hash = asx::sha256_hex(asx::canonical_json(trace_doc));
            return result;
        }
        const auto edge_it = state_it->second.find(token);
        if (edge_it == state_it->second.end()) {
            result.final_state = state;
            result.pass = false;
            asx::Value trace_doc = asx::Value::make_object();
            trace_doc.object_value["@state_trace"] = asx::Value::make_array();
            for (const auto& entry : result.trace) trace_doc.object_value["@state_trace"].array_value.push_back(asx::Value::make_string(entry));
            result.trace_hash = asx::sha256_hex(asx::canonical_json(trace_doc));
            return result;
        }
        state = edge_it->second;
        result.trace.push_back(state);
    }

    result.final_state = state;
    result.pass = dfa.accept.count(state) > 0;
    asx::Value trace_doc = asx::Value::make_object();
    trace_doc.object_value["@state_trace"] = asx::Value::make_array();
    for (const auto& entry : result.trace) trace_doc.object_value["@state_trace"].array_value.push_back(asx::Value::make_string(entry));
    result.trace_hash = asx::sha256_hex(asx::canonical_json(trace_doc));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: mx2lex_oracle <dfa.asx> <request.asx>\n";
        return 2;
    }

    try {
        const DfaDocument dfa = load_dfa(argv[1]);
        const OracleRequest request = load_request(argv[2]);
        const OracleResult result = run_oracle(dfa, request);

        std::cout << "{\n";
        std::cout << "  \"@kind\": \"mx2lex.oracle.result.v1\",\n";
        std::cout << "  \"@ok\": true,\n";
        std::cout << "  \"@dfa_id\": \"" << escape_json(dfa.id) << "\",\n";
        std::cout << "  \"@request_id\": \"" << escape_json(request.id) << "\",\n";
        std::cout << "  \"@dfa_hash\": \"" << dfa.document_hash << "\",\n";
        std::cout << "  \"@request_hash\": \"" << request.document_hash << "\",\n";
        std::cout << "  \"@decision\": \"" << (result.pass ? "PASS" : "FAIL") << "\",\n";
        std::cout << "  \"@final_state\": \"" << escape_json(result.final_state) << "\",\n";
        std::cout << "  \"@trace_hash\": \"" << result.trace_hash << "\",\n";
        std::cout << "  \"@trace\": [";
        for (std::size_t i = 0; i < result.trace.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << "\"" << escape_json(result.trace[i]) << "\"";
        }
        std::cout << "]\n";
        std::cout << "}\n";
        return result.pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "{\n";
        std::cerr << "  \"@kind\": \"mx2lex.oracle.result.v1\",\n";
        std::cerr << "  \"@ok\": false,\n";
        std::cerr << "  \"@error\": \"" << escape_json(error.what()) << "\"\n";
        std::cerr << "}\n";
        return 1;
    }
}
