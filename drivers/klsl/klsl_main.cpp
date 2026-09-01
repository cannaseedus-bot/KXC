/**
 * klsl_main.cpp
 * Command-line KLSL compiler driver.
 *
 * Usage:
 *   klslc input.klsl                 → writes input.hlsl
 *   klslc input.klsl output.hlsl     → explicit output path
 *   klslc input.klsl --xvm out.bin   → XVM bytecode
 */

#include "klsl_compiler.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "KLSL Compiler v0.1\n"
                     "Usage: klslc <input.klsl> [output.hlsl] [--xvm output.bin]\n";
        return 1;
    }

    std::string inFile  = argv[1];
    std::string outHLSL;
    std::string outXVM;
    bool emitXVM = false;

    std::cerr << "KLSLC BINARY FROM 2026-08-31-1852-DEBUG\n";
    std::cerr << "DEBUG: argc=" << argc << "\n";
    for (int j = 0; j < argc; ++j) {
        std::cerr << "DEBUG: argv[" << j << "]=\"" << argv[j] << "\"\n";
    }

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--xvm") == 0 && i + 1 < argc) {
            outXVM  = argv[++i];
            emitXVM = true;
            std::cerr << "DEBUG: Found --xvm, outXVM=\"" << outXVM << "\"\n";
        } else {
            outHLSL = argv[i];
            std::cerr << "DEBUG: Found outHLSL=\"" << outHLSL << "\"\n";
        }
    }

    // Derive default HLSL output name
    if (outHLSL.empty()) {
        outHLSL = inFile;
        size_t dot = outHLSL.rfind('.');
        if (dot != std::string::npos) outHLSL = outHLSL.substr(0, dot);
        outHLSL += ".hlsl";
    }

    // Read source
    std::ifstream ifs(inFile);
    if (!ifs) {
        std::cerr << "klslc: cannot open " << inFile << "\n";
        return 1;
    }
    std::ostringstream src;
    src << ifs.rdbuf();

    // Compile
    klsl::CompileOptions opts;
    opts.emitHLSL = true;
    opts.emitXVM  = emitXVM;

    klsl::CompileResult res = klsl::compile(src.str(), inFile, opts);
    if (!res.ok) {
        std::cerr << "klslc: " << res.errorMsg << "\n";
        return 1;
    }

    // Write HLSL
    {
        std::ofstream ofs(outHLSL);
        if (!ofs) { std::cerr << "klslc: cannot write " << outHLSL << "\n"; return 1; }
        ofs << res.hlsl;
        std::cout << "KLSL → HLSL  " << inFile << "  →  " << outHLSL << "\n";
    }

    // Write XVM bytecode
    if (emitXVM) {
        std::cerr << "DEBUG: emitXVM=true, res.xvm.size()=" << res.xvm.size() << "\n";
        if (!res.xvm.empty()) {
            std::ofstream ofs(outXVM, std::ios::binary);
            if (!ofs) { std::cerr << "klslc: cannot write " << outXVM << "\n"; return 1; }
            ofs.write(reinterpret_cast<const char*>(res.xvm.data()), (std::streamsize)res.xvm.size());
            std::cout << "KLSL → XVM   " << inFile << "  →  " << outXVM
                      << "  (" << res.xvm.size() << " bytes)\n";
        } else {
            std::cerr << "klslc: XVM bytecode is empty (compilation issue)\n";
        }
    }

    return 0;
}
