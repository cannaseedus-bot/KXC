#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "asx_parser.h"
#include "asx_value.h"

namespace {

struct Rule {
    std::string lhs;
    std::string symbol;
    std::string next;
    bool accept = false;
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
            rule.accept = false;
        } else {
            throw std::runtime_error("rule must be right-linear with rhs size 1 or 2");
        }
        out.push_back(rule);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: mx2lex_compile <grammar.asx>\n";
        return 2;
    }

    try {
        const asx::Value document = asx::parse_file(argv[1]);
        const asx::Value* root = asx::get_child(document, "mx2lex.grammar");
        if (root == nullptr || !root->is_object()) throw std::runtime_error("missing @@mx2lex.grammar block");

        const std::string grammar_id = asx::require_string(*root, "@id");
        const std::string start = asx::require_string(*root, "start");
        const auto terminals = require_string_array(*root, "terminals");
        const auto nonterminals = require_string_array(*root, "nonterminals");
        const auto rules = load_rules(*root);

        std::set<std::string> terminal_set(terminals.begin(), terminals.end());
        std::set<std::string> nonterminal_set(nonterminals.begin(), nonterminals.end());
        std::set<std::string> accept;
        std::map<std::string, std::map<std::string, std::string>> transitions;

        if (nonterminal_set.count(start) == 0) throw std::runtime_error("start symbol is not declared as nonterminal");

        for (const auto& rule : rules) {
            if (nonterminal_set.count(rule.lhs) == 0) throw std::runtime_error("rule lhs is not declared nonterminal: " + rule.lhs);
            if (terminal_set.count(rule.symbol) == 0) throw std::runtime_error("rule symbol is not declared terminal: " + rule.symbol);
            if (!rule.accept && nonterminal_set.count(rule.next) == 0) {
                throw std::runtime_error("rule next state is not declared nonterminal: " + rule.next);
            }
            auto& state_transitions = transitions[rule.lhs];
            auto existing = state_transitions.find(rule.symbol);
            const std::string target = rule.accept ? rule.lhs : rule.next;
            if (existing != state_transitions.end() && existing->second != target) {
                throw std::runtime_error("grammar is nondeterministic for state/symbol pair: " + rule.lhs + " / " + rule.symbol);
            }
            state_transitions[rule.symbol] = target;
            if (rule.accept) accept.insert(rule.lhs);
        }

        std::vector<std::string> states(nonterminal_set.begin(), nonterminal_set.end());
        std::sort(states.begin(), states.end());

        std::cout << "@@mx2lex.dfa\n";
        std::cout << "  @id: \"" << escape_json(grammar_id + ".dfa") << "\"\n";
        std::cout << "  @version: \"1.0.0\"\n";
        std::cout << "  start: \"" << escape_json(start) << "\"\n";
        std::cout << "  accept: [";
        {
            bool first = true;
            for (const auto& state : accept) {
                if (!first) std::cout << ", ";
                first = false;
                std::cout << "\"" << escape_json(state) << "\"";
            }
        }
        std::cout << "]\n\n";
        std::cout << "  transitions\n";
        for (const auto& state : states) {
            const auto it = transitions.find(state);
            if (it == transitions.end()) continue;
            for (const auto& edge : it->second) {
                std::cout << "    -\n";
                std::cout << "      from: \"" << escape_json(state) << "\"\n";
                std::cout << "      sym: \"" << escape_json(edge.first) << "\"\n";
                std::cout << "      to: \"" << escape_json(edge.second) << "\"\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\n";
        std::cerr << "  \"@kind\": \"mx2lex.compile.result.v1\",\n";
        std::cerr << "  \"@ok\": false,\n";
        std::cerr << "  \"@error\": \"" << escape_json(error.what()) << "\"\n";
        std::cerr << "}\n";
        return 1;
    }
}
