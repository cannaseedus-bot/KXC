#include "tokenizer.h"
#include <algorithm>
#include <cctype>
#include <sstream>

TinyTokenizer::TinyTokenizer() {}

static std::string strip_punct(const std::string &s){
    std::string out;
    for (char c: s) if (!std::ispunct((unsigned char)c)) out.push_back(c);
    return out;
}

std::vector<std::string> TinyTokenizer::tokenize(const std::string &text){
    std::istringstream iss(text);
    std::string w;
    std::vector<std::string> toks;
    while (iss >> w){
        std::string s = strip_punct(w);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        if (!s.empty()) toks.push_back(s);
    }
    return toks;
}

std::string TinyTokenizer::lemmatize(const std::string &token){
    // ultra-simple rules: remove common verb endings
    if (token.size()>4){
        if (token.substr(token.size()-3)=="ing") return token.substr(0, token.size()-3);
        if (token.substr(token.size()-2)=="ed") return token.substr(0, token.size()-2);
    }
    // plural 's'
    if (token.size()>2 && token.back()=='s') return token.substr(0, token.size()-1);
    return token;
}
