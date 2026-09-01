# SKELETON

## Project
- **Name:** KHANARY
- **Repository Root:** `C:\Users\canna\_khanary_inspect`
- **Owner:** _(fill)_
- **Phase:** _(fill)_

## Goal
- **Primary objective:** _(fill)_
- **Current focus:** _(fill)_
- **Success criteria:** _(fill)_

## Runtime Paths
- **Canonical runtime:** `dist\kuhul-es`
- **Legacy/prototype references:** `.Powernaut-v1.0.0\kuhul`

## Core Entrypoints
- **Launcher:** `START-SERVERS.bat`
- **Primary server (dist lane):** `dist\khanary-server\khanary-server.exe`
- **Optional server (build lane):** `khanary-llama-build\llama.cpp\build\bin\Release\llama-server.exe`

## Build Entrypoints
1. `bridges\ggml-xcfe\build_xcfe.bat`
2. `llama-build.bat`
3. `build-khanary.bat`

## Key Runtime Files
- `dist\khanary-server\ggml-xcfe.dll` — XCFE backend plugin
- `dist\khanary-server\xcfe_gl_ops.dll` — OpenGL compute ops bridge
- `dist\khanary-server\drivers\ggml-xcfe.dll` — mirrored backend
- `dist\khanary-server\drivers\xcfe_gl_ops.dll` — mirrored GL ops

## KAST Example Artifact
- `dist\kuhul-es\examples\trained_skeleton.json` — trained KAST skeleton sample (`protocol: kast/1`)
- Top-level keys: `protocol`, `source_kind`, `nodes`, `edges`, `physics`, `loss_history`, `config`
- Use this as a reference structure for skeleton/schema and Q/A retrieval tests.

## Memory + Event Stream
- **State projection:** `.asx_memory.json`
- **Event log (append-only):** `.asx_memory.events.jsonl`
- **Notes:** keep event records short, factual, and path-specific.

## Open Tasks
- [ ] _(task 1)_
- [ ] _(task 2)_
- [ ] _(task 3)_

## Decisions
- **Dxxx:** _(decision summary)_
- **Dyyy:** _(decision summary)_

## Verification Checklist
1. Build artifacts exist in `dist\khanary-server\`.
2. Launcher starts expected server lane.
3. Log markers match expected runtime path.
4. No regression in active endpoint behavior.

## Notes
- _(freeform)_
