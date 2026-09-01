#pragma once
//
// khanary_driver.h — Native driver DLL for TaskEngine + DAG + provider dispatch
//
// Flat C ABI. Loaded by kuhul-server via ffi-napi/koffi.
// Compiles with MSVC (no Emscripten, no WASM).
//

#ifdef KHANARY_DRIVER_EXPORTS
#define KHANARY_API __declspec(dllexport)
#else
#define KHANARY_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ── Lifetime ──────────────────────────────────────────────────────────

/// Create driver with providers JSON (array of {id, library, available}).
/// Returns opaque handle, or NULL on failure.
KHANARY_API void* kd_create(const char* providers_json);

/// Destroy driver handle.
KHANARY_API void kd_destroy(void* driver);

// ── Load ──────────────────────────────────────────────────────────────

/// Load TaskList JSON string. Writes error to error_buf (max error_buf_size)
/// on failure. Returns 1 on success, 0 on failure.
KHANARY_API int kd_load_tasks(void* driver, const char* tasklist_json,
                               char* error_buf, int error_buf_size);

// ── Plan ──────────────────────────────────────────────────────────────

/// Run plan() and return results as JSON string.
/// Caller frees with kd_free_string().
/// Format: [{"id":"...","provider":"...","status":"...","detail":"..."},...]
KHANARY_API char* kd_plan(void* driver);

// ── Run ───────────────────────────────────────────────────────────────

/// Run plan + execute, return results as JSON string.
/// Caller frees with kd_free_string().
KHANARY_API char* kd_run(void* driver);

// ── Dispatch ──────────────────────────────────────────────────────────

/// Dispatch a single task via MCP JSON-RPC (kuhul_task_boss protocol).
/// task_json: {"id":"...","action":"...","provider":"...","description":"..."}
/// Returns JSON result: {"id":"...","status":"...","detail":"..."}
/// Caller frees with kd_free_string().
KHANARY_API char* kd_dispatch(void* driver, const char* task_json);

// ── Query ─────────────────────────────────────────────────────────────

KHANARY_API int kd_task_count(void* driver);

KHANARY_API void kd_free_string(char* str);

#ifdef __cplusplus
}
#endif
