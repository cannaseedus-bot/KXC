#pragma once

#include <string>
#include <vector>
#include "tokenizer.h"


struct SemanticArgument {
    std::string role;
    std::string entity;
};

struct SemanticClause {
    std::string predicate;
    std::vector<SemanticArgument> arguments;
    std::string tense;
    std::string polarity;
};

class SemanticKernel {
public:
    SemanticKernel();
    // detect language (very simple heuristics)
    std::pair<std::string,double> detect_language(const std::string &text);
    // parse utterance into a simple semantic clause
    SemanticClause parse_to_semantics(const std::string &utterance, const std::string &lang="english");
    // write IR (JSON) to file
    bool write_ir_json(const SemanticClause &clause, const std::string &path);
    // write IR (XML) to file
    bool write_ir_xml(const SemanticClause &clause, const std::string &path);
    // write IR (Protobuf binary) to file (requires Protobuf & CMake generation)
    bool write_ir_protobuf(const SemanticClause &clause, const std::string &path);
    // query XML IR with a simple XPath-like path (supports /a/b and predicates like element[child='value'])
    bool query_ir_xml(const std::string &xml_path, const std::string &query, std::vector<std::string> &results);
    // express semantics into target language (surface form)
    std::string express(const SemanticClause &clause, const std::string &target_lang="english");
};
