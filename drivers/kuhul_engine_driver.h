#pragma once
//
// kuhul_engine_driver.h — Native driver DLL for kuhul_engine
//
// Wraps model inference + Atomic DOM + chat completion in a flat C ABI.
// Loaded by kuhul-server via ffi-napi. Compiles with MSVC.
//
// Replaces the subprocess model (kuhul_engine.exe --serve :17480) with
// in-process dispatch when loaded alongside khanary_driver.dll and
// khanary_glyph_driver.dll.
//

#ifdef KUHUL_ENGINE_EXPORTS
#define KUHUL_ENGINE_API __declspec(dllexport)
#else
#define KUHUL_ENGINE_API __declspec(dllimport)
#endif

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Engine lifecycle ──────────────────────────────────────────────────

/// Create engine instance. Returns opaque handle.
/// config_json: {"model_path":"...","n_ctx":1024,"n_threads":4,"ngl":0}
KUHUL_ENGINE_API void* ke_create(const char* config_json);

/// Destroy engine instance.
KUHUL_ENGINE_API void ke_destroy(void* engine);

/// Load a model GGUF. Returns 1 on success, 0 on failure.
/// Writes error to error_buf on failure.
KUHUL_ENGINE_API int ke_load_model(void* engine, const char* model_path,
                                    char* error_buf, int error_buf_size);

// ── Atomic DOM ────────────────────────────────────────────────────────

/// Load an Atomic DOM manifest. Returns 1 on success.
/// manifest_path: path to atomic.manifest.json
KUHUL_ENGINE_API int ke_load_dom(void* engine, const char* manifest_path,
                                  char* error_buf, int error_buf_size);

/// Get the tool registry from the loaded DOM as JSON.
/// Caller frees with ke_free_string.
KUHUL_ENGINE_API char* ke_get_tools(void* engine);

/// Get the model's NPC persona from the DOM.
/// Caller frees with ke_free_string.
KUHUL_ENGINE_API char* ke_get_persona(void* engine);

/// Check if execution gating is active for this DOM.
KUHUL_ENGINE_API int ke_is_execution_gated(void* engine);

// ── Chat completion ───────────────────────────────────────────────────

/// Run a single chat completion turn.
/// messages_json: [{"role":"user","content":"..."}] or
///                [{"role":"system","content":"..."},{"role":"user","content":"..."}]
/// Returns the assistant's response as JSON:
///   {"role":"assistant","content":"...","tool_calls":[...]}
/// Caller frees with ke_free_string.
KUHUL_ENGINE_API char* ke_chat(void* engine, const char* messages_json);

/// Stream a chat completion. Returns a token iterator handle.
/// Use ke_next_token() to pull tokens one at a time.
KUHUL_ENGINE_API void* ke_chat_stream(void* engine, const char* messages_json);

/// Get the next token from a streaming chat. Returns NULL when done.
/// Caller frees each token with ke_free_string.
KUHUL_ENGINE_API char* ke_next_token(void* stream);

/// Cancel a streaming chat.
KUHUL_ENGINE_API void ke_cancel_stream(void* stream);

// ── Status ────────────────────────────────────────────────────────────

/// Get engine status as JSON: {"model_loaded":true,"dom_loaded":true,...}
/// Caller frees with ke_free_string.
KUHUL_ENGINE_API char* ke_status(void* engine);

/// Check if a model is currently loaded.
KUHUL_ENGINE_API int ke_is_model_loaded(void* engine);

// ── WWA packaging (zipcontainer → .wwa bundle) ────────────────────────

/// Package an app folder into a .wwa zip container (deterministic STORE
/// compression, same layout WWAHost.exe expects: manifest.json + entry html).
/// Returns 1 on success, 0 on failure (error in error_buf).
KUHUL_ENGINE_API int ke_package_wwa(const char* app_root,
                                    const char* out_wwa_path,
                                    char* error_buf, int error_buf_size);

/// Launch a .wwa bundle (or app folder) through WWAHost.exe (detached).
/// Returns 1 on success, 0 on failure (error in error_buf).
KUHUL_ENGINE_API int ke_launch_wwa(const char* wwa_path,
                                   char* error_buf, int error_buf_size);

/// Check whether WWAHost.exe + WwaExt.dll + WwaApi.dll are present.
/// Returns 1 if the full WWA runtime is available.
KUHUL_ENGINE_API int ke_wwa_runtime_available(void);

// ── Utility ───────────────────────────────────────────────────────────

KUHUL_ENGINE_API void ke_free_string(char* str);

#ifdef __cplusplus
}
#endif
