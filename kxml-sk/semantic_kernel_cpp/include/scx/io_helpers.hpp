// ---------------------------------------------------------------
// io_helpers.hpp – tiny file / path utilities
// ---------------------------------------------------------------
#pragma once
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

namespace scx::io {

inline std::string readFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read file: " + p.string());
    std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}

inline void writeFile(const std::filesystem::path& p, const std::string& data) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write file: " + p.string());
    out << data;
    out.flush();
}

inline std::filesystem::path userBase() {
    const char* home = std::getenv("USERPROFILE");
    if (!home) throw std::runtime_error("$USERPROFILE not set");
    return std::filesystem::path(home) / ".kuhul-v1" / "micronaut-coder";
}

inline std::filesystem::path grammarDir(const std::string& lang) {
    return userBase() / "grammars" / lang;
}

inline std::filesystem::path cacheDir() { return userBase() / "cache" / "web"; }

inline std::filesystem::path logFile() { return userBase() / "logs" / "micronaut_coder.log"; }

inline void log(const std::string& msg) {
    try {
        std::filesystem::create_directories(logFile().parent_path());
        std::ofstream out(logFile(), std::ios::app);
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        out << "[" << now << "] " << msg << "\n";
    } catch (...) {}
}
}
