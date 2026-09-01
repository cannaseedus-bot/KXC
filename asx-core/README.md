# Native ASX Core

This directory contains the native parser/hash/canonicalization layer for the
current `.asx` stack:

- `asx_parser.*`: indentation-based parser for the envelope/control/schema subset
- `asx_canonical.*`: deterministic JSON serializer with sorted object keys
- `sha256.*`: real SHA-256 implementation for canonical document and target-file binding
- `asx_verifier.cpp`: machine-readable verifier for `@@module.envelope` inputs
- `mx2lex_compile.cpp`: right-linear grammar compiler from `@@mx2lex.grammar` to `@@mx2lex.dfa`
- `mx2lex_oracle.cpp`: machine-readable DFA oracle for `@@mx2lex.dfa` and `@@mx2lex.oracle.request`
- `mx2lex_vector_runner.cpp`: native batch runner for grammar/request PASS/FAIL vectors
- `scxqdds.cpp`: native SQDS (SCXQ-DDS) decode + validate (varints + CRC32 + reconstruction)
- `scxqdds_decode.cpp`: machine-readable SQDS decoder CLI
- `scxqdds_selftest.cpp`: deterministic SQDS selftest (no external inputs)
- `scxqdds_vector_runner.cpp`: native batch runner for SQDS golden vectors (hex fixtures)

The parser is intentionally narrow. It covers the current lawful native slice:

- `@@block`
- nested objects by indentation
- arrays via `- item`
- scalar strings, booleans, numbers, `null`

It does not try to execute or interpret anything.

Current native entrypoints:

- `artifacts/native-asx/asx_verifier.exe codex/examples/example.module.asx`
- `artifacts/native-asx/mx2lex_compile.exe codex/examples/mx2lex.grammar.asx > artifacts/native-asx/mx2lex.compiled.dfa.asx`
- `artifacts/native-asx/mx2lex_oracle.exe codex/examples/mx2lex.dfa.asx codex/examples/mx2lex.oracle.request.asx`
- `artifacts/native-asx/mx2lex_vector_runner.exe codex/examples/mx2lex.vectors.asx`
- `artifacts/native-asx/scxqdds_selftest.exe`
- `artifacts/native-asx/scxqdds_decode.exe <file.sqds>`
- `artifacts/native-asx/scxqdds_vector_runner.exe codex/examples/scxqdds.vectors.asx`
- `artifacts/native-asx/scxq2_vector_runner.exe codex/examples/scxq2.vectors.asx`

The MX2LEX path now supports both authored DFA contracts and compiled grammar
contracts. The oracle ABI stays fixed on `@@mx2lex.dfa`, and the vector runner
compiles grammar in-memory before executing the batch.

## Build environment (MSVC)

`scripts/build-asx-native-msvc.bat` prefers environment variables to avoid hardcoded toolchain paths:

- `ASX_VSDEV` → full path to `VsDevCmd.bat` (e.g., `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat`)
- `ASX_MSVCROOT` → MSVC toolset root (e.g., `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.3x.xxxxx`)
- `ASX_CL` (optional) → explicit `cl.exe` path

If these are unset, the script will try `vswhere` discovery; set them explicitly on hosts without VS installers.
