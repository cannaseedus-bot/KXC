#pragma once

#include <string>
#include "nlohmann/json.hpp"

struct SemanticReaderInput {
    std::string source_name;
    std::string text;
    float activation_threshold = 0.35f;
};

class SemanticReader {
public:
    nlohmann::json read(const SemanticReaderInput& input) const;
};

bool write_semantic_reader_report(const std::string& input_path,
                                  const std::string& output_path,
                                  float activation_threshold = 0.35f);

bool write_semantic_jsonl_absorb_report(const std::string& input_path,
                                        const std::string& output_path,
                                        float activation_threshold = 0.35f);
