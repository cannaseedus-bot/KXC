#include <algorithm>
#include <cctype>
#include <filesystem>
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

struct Rule {
    std::string lhs;
    std::string symbol;
    std::string next;
    bool accept = false;
};

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
    std::string hash;
};

struct OracleRequest {
    std::string id;
    std::vector<std::string> tokens;
    std::string hash;
};

struct VectorCase {
    std::string id;
    std::string request_path;
    std::string expected;
};

struct OracleResult {
    std::string decision;
    std::string final_state;
    std::vector<std::string> trace;
    std::string trace_hash;
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

namespace fs = std::filesystem;

std::string to_lower_ascii(std::string value) {
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

fs::path safe_resolve_under(const fs::path& base_dir, const std::string& relative_path) {
    const fs::path base = fs::weakly_canonical(base_dir);
    const fs::path candidate = fs::weakly_canonical(base / fs::path(relative_path));

    // Prevent traversal outside the vectors directory.
    const std::string base_str = to_lower_ascii(base.lexically_normal().string());
    const std::string cand_str = to_lower_ascii(candidate.lexically_normal().string());
    if (cand_str.rfind(base_str, 0) != 0) throw std::runtime_error("path escapes vectors directory: " + relative_path);
    if (cand_str.size() > base_str.size()) {
        const char boundary = cand_str[base_str.size()];
        if (boundary != '\\' && boundary != '/') {
            throw std::runtime_error("path escapes vectors directory: " + relative_path);
        }
    }
    return candidate;
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

std::vector<Rule> load_rules(const asx::Value& root) {
    const asx::Value* rules = asx::get_path(root, "rules");
    if (rules == nullptr || !rules->is_array()) throw std::runtime_error("expected rules array");
    std::vector<Rule> out;
    for (const auto& item : rules->array_value) {
        if (!item.is_object()) throw std::runtime_error("rule entry must be object");
        Rule rule;
        rule.lhs = asx::require_string(item, "lhs");
        const auto rhs = require_string_array(item, "rhs");
        if (rhs.size() == 1U) {
            rule.symbol = rhs[0];
            rule.accept = true;
        } else if (rhs.size() == 2U) {
            rule.symbol = rhs[0];
            rule.next = rhs[1];
        } else {
            throw std::runtime_error("rule must be right-linear with rhs size 1 or 2");
        }
        out.push_back(rule);
    }
    return out;
}

DfaDocument compile_grammar(const std::string& path) {
    const asx::Value document = asx::parse_file(path);
    const asx::Value* root = asx::get_child(document, "mx2lex.grammar");
    if (root == nullptr || !root->is_object()) throw std::runtime_error("missing @@mx2lex.grammar block");

    const std::string grammar_id = asx::require_string(*root, "@id");
    const std::string start = asx::require_string(*root, "start");
    const auto terminals = require_string_array(*root, "terminals");
    const auto nonterminals = require_string_array(*root, "nonterminals");
    const auto rules = load_rules(*root);

    std::set<std::string> terminal_set(terminals.begin(), terminals.end());
    std::set<std::string> nonterminal_set(nonterminals.begin(), nonterminals.end());
    std::map<std::string, std::map<std::string, std::string>> transitions;
    std::set<std::string> accept;

    if (nonterminal_set.count(start) == 0) throw std::runtime_error("start symbol is not declared as nonterminal");

    for (const auto& rule : rules) {
        if (nonterminal_set.count(rule.lhs) == 0) throw std::runtime_error("rule lhs is not declared nonterminal: " + rule.lhs);
        if (terminal_set.count(rule.symbol) == 0) throw std::runtime_error("rule symbol is not declared terminal: " + rule.symbol);
        if (!rule.accept && nonterminal_set.count(rule.next) == 0) {
            throw std::runtime_error("rule next state is not declared nonterminal: " + rule.next);
        }
        const std::string target = rule.accept ? rule.lhs : rule.next;
        auto& state_transitions = transitions[rule.lhs];
        auto existing = state_transitions.find(rule.symbol);
        if (existing != state_transitions.end() && existing->second != target) {
            throw std::runtime_error("grammar is nondeterministic for state/symbol pair: " + rule.lhs + " / " + rule.symbol);
        }
        state_transitions[rule.symbol] = target;
        if (rule.accept) accept.insert(rule.lhs);
    }

    DfaDocument out;
    out.id = grammar_id + ".dfa";
    out.start = start;
    out.accept = accept;
    for (const auto& state : nonterminal_set) {
        const auto it = transitions.find(state);
        if (it == transitions.end()) continue;
        for (const auto& edge : it->second) {
            out.transitions.push_back({state, edge.first, edge.second});
        }
    }
    std::sort(out.transitions.begin(), out.transitions.end(), [](const Transition& a, const Transition& b) {
        if (a.from != b.from) return a.from < b.from;
        if (a.sym != b.sym) return a.sym < b.sym;
        return a.to < b.to;
    });

    asx::Value dfa_doc = asx::Value::make_object();
    asx::Value dfa_root = asx::Value::make_object();
    dfa_root.object_value["@id"] = asx::Value::make_string(out.id);
    dfa_root.object_value["@version"] = asx::Value::make_string("1.0.0");
    dfa_root.object_value["start"] = asx::Value::make_string(out.start);
    dfa_root.object_value["accept"] = asx::Value::make_array();
    for (const auto& state : out.accept) dfa_root.object_value["accept"].array_value.push_back(asx::Value::make_string(state));
    dfa_root.object_value["transitions"] = asx::Value::make_array();
    for (const auto& transition : out.transitions) {
      asx::Value item = asx::Value::make_object();
      item.object_value["from"] = asx::Value::make_string(transition.from);
      item.object_value["sym"] = asx::Value::make_string(transition.sym);
      item.object_value["to"] = asx::Value::make_string(transition.to);
      dfa_root.object_value["transitions"].array_value.push_back(item);
    }
    dfa_doc.object_value["mx2lex.dfa"] = dfa_root;
    out.hash = asx::sha256_hex(asx::canonical_json(dfa_doc));
    return out;
}

OracleRequest load_request(const std::string& path) {
    const asx::Value document = asx::parse_file(path);
    const asx::Value* root = asx::get_child(document, "mx2lex.oracle.request");
    if (root == nullptr || !root->is_object()) throw std::runtime_error("missing @@mx2lex.oracle.request block");
    OracleRequest out;
    out.id = asx::require_string(*root, "@id");
    out.tokens = require_string_array(*root, "tokens");
    out.hash = asx::sha256_hex(asx::canonical_json(document));
    return out;
}

OracleResult run_oracle(const DfaDocument& dfa, const OracleRequest& request) {
    std::map<std::string, std::map<std::string, std::string>> transitions;
    for (const auto& transition : dfa.transitions) {
        transitions[transition.from][transition.sym] = transition.to;
    }

    OracleResult result;
    std::string state = dfa.start;
    result.trace.push_back(state);

    for (const auto& token : request.tokens) {
        const auto state_it = transitions.find(state);
        if (state_it == transitions.end()) {
            result.decision = "FAIL";
            result.final_state = state;
            asx::Value trace_doc = asx::Value::make_object();
            trace_doc.object_value["@state_trace"] = asx::Value::make_array();
            for (const auto& entry : result.trace) trace_doc.object_value["@state_trace"].array_value.push_back(asx::Value::make_string(entry));
            result.trace_hash = asx::sha256_hex(asx::canonical_json(trace_doc));
            return result;
        }
        const auto edge_it = state_it->second.find(token);
        if (edge_it == state_it->second.end()) {
            result.decision = "FAIL";
            result.final_state = state;
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
    result.decision = dfa.accept.count(state) > 0 ? "PASS" : "FAIL";
    asx::Value trace_doc = asx::Value::make_object();
    trace_doc.object_value["@state_trace"] = asx::Value::make_array();
    for (const auto& entry : result.trace) trace_doc.object_value["@state_trace"].array_value.push_back(asx::Value::make_string(entry));
    result.trace_hash = asx::sha256_hex(asx::canonical_json(trace_doc));
    return result;
}

std::vector<VectorCase> load_cases(const asx::Value& root) {
    const asx::Value* cases = asx::get_path(root, "cases");
    if (cases == nullptr || !cases->is_array()) throw std::runtime_error("expected cases array");
    std::vector<VectorCase> out;
    for (const auto& item : cases->array_value) {
        if (!item.is_object()) throw std::runtime_error("case entry must be object");
        out.push_back({
            asx::require_string(item, "@id"),
            asx::require_string(item, "request"),
            asx::require_string(item, "expect"),
        });
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: mx2lex_vector_runner <vectors.asx>\n";
        return 2;
    }

    try {
        const fs::path vector_path = fs::path(argv[1]);
        const fs::path vector_dir = vector_path.has_parent_path() ? vector_path.parent_path() : fs::path(".");

        const asx::Value document = asx::parse_file(vector_path.string());
        const asx::Value* root = asx::get_child(document, "mx2lex.vectors");
        if (root == nullptr || !root->is_object()) throw std::runtime_error("missing @@mx2lex.vectors block");

        const std::string vector_id = asx::require_string(*root, "@id");
        const std::string grammar_path = asx::require_string(*root, "grammar");
        const auto cases = load_cases(*root);
        const fs::path resolved_grammar = safe_resolve_under(vector_dir, grammar_path);
        const DfaDocument dfa = compile_grammar(resolved_grammar.string());

        int pass_count = 0;
        int fail_count = 0;
        int mismatch_count = 0;
        std::vector<std::string> detail_hash_parts;

        std::cout << "{\n";
        std::cout << "  \"@kind\": \"mx2lex.vector.summary.v1\",\n";
        std::cout << "  \"@ok\": true,\n";
        std::cout << "  \"@id\": \"" << escape_json(vector_id) << "\",\n";
        std::cout << "  \"@dfa_id\": \"" << escape_json(dfa.id) << "\",\n";
        std::cout << "  \"@dfa_hash\": \"" << dfa.hash << "\",\n";
        std::cout << "  \"@cases\": [\n";

        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& entry = cases[i];
            const fs::path resolved_request = safe_resolve_under(vector_dir, entry.request_path);
            const OracleRequest request = load_request(resolved_request.string());
            const OracleResult result = run_oracle(dfa, request);
            const bool matched = (result.decision == entry.expected);
            if (result.decision == "PASS") ++pass_count; else ++fail_count;
            if (!matched) ++mismatch_count;
            detail_hash_parts.push_back(entry.id + "|" + request.hash + "|" + result.trace_hash + "|" + result.decision + "|" + entry.expected);

            std::cout << "    {\"@id\":\"" << escape_json(entry.id)
                      << "\",\"@request_id\":\"" << escape_json(request.id)
                      << "\",\"@decision\":\"" << result.decision
                      << "\",\"@expected\":\"" << escape_json(entry.expected)
                      << "\",\"@matched\":" << (matched ? "true" : "false")
                      << ",\"@trace_hash\":\"" << result.trace_hash << "\""
                      << ",\"@trace\":[";
            for (std::size_t t = 0; t < result.trace.size(); ++t) {
                if (t > 0) std::cout << ",";
                std::cout << "\"" << escape_json(result.trace[t]) << "\"";
            }
            std::cout << "]}";
            if (i + 1 != cases.size()) std::cout << ",";
            std::cout << "\n";
        }

        std::sort(detail_hash_parts.begin(), detail_hash_parts.end());
        std::string summary_seed;
        for (const auto& part : detail_hash_parts) summary_seed += part + "\n";
        const std::string summary_hash = asx::sha256_hex(summary_seed);
        const std::string final_verdict = mismatch_count == 0 ? "PASS" : "FAIL";

        std::cout << "  ],\n";
        std::cout << "  \"@counts\": {\"pass\": " << pass_count << ", \"fail\": " << fail_count << ", \"mismatch\": " << mismatch_count << "},\n";
        std::cout << "  \"@verdict\": \"" << final_verdict << "\",\n";
        std::cout << "  \"@summary_hash\": \"" << summary_hash << "\"\n";
        std::cout << "}\n";
        return mismatch_count == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "{\n";
        std::cerr << "  \"@kind\": \"mx2lex.vector.summary.v1\",\n";
        std::cerr << "  \"@ok\": false,\n";
        std::cerr << "  \"@error\": \"" << escape_json(error.what()) << "\"\n";
        std::cerr << "}\n";
        return 2;
    }
}
