// ============================================================================
// tokenizer.h - Tiny Tokenizer for Semantic Kernel (ASX v0.7)
// ============================================================================

#pragma once

#include <vector>
#include <string>

class TinyTokenizer {
public:
    TinyTokenizer();
    
    // Split text into tokens and lowercases
    std::vector<std::string> tokenize(const std::string &text);
    
    // Very simple lemmatization (removes ing, ed, s)
    std::string lemmatize(const std::string &token);
};
