#pragma once
//
// dag_abi.h — flat C ABI for the K'UHUL DAG scheduler DLL.
//
// Build: cl /LD /EHsc /O2 /std:c++17 dag_abi.cpp DAG.cpp /Fe:dag.dll
//
// Loaded by kxc.exe, kuhul-server, and micronaut-v4 via ffi/koffi.
//

#ifdef DAG_EXPORTS
#define DAG_API __declspec(dllexport)
#else
#define DAG_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ── Lifetime ──────────────────────────────────────────────────────────

/// Create an empty scheduler context. Returns opaque handle.
DAG_API void* dag_create(void);

/// Destroy a scheduler context.
DAG_API void dag_destroy(void* ctx);

// ── Task loading ──────────────────────────────────────────────────────

/// Load tasks from a JSON array string.
/// Expected format:
///   [{"id":"a","depends_on":["b"]}, ...]
/// The field "depends_on" or "dependsOn" is accepted.
/// Returns 1 on success, 0 on failure (error written to error_buf).
DAG_API int dag_load_tasks(void* ctx, const char* tasks_json,
                           char* error_buf, int error_buf_size);

// ── Scheduling ────────────────────────────────────────────────────────

/// Compute a topological ordering.
/// Returns JSON: {"ordered":["a","b",...],"error":""}
/// Caller frees the returned string with dag_free_string.
DAG_API char* dag_schedule(void* ctx);

/// Convenience: schedule a JSON task array in one call.
/// tasks_json is the same format accepted by dag_load_tasks.
/// Returns the same JSON format as dag_schedule.
DAG_API char* dag_schedule_json(const char* tasks_json,
                                char* error_buf, int error_buf_size);

// ── Validation / inspection ───────────────────────────────────────────

/// Return the number of tasks loaded in the context.
DAG_API int dag_task_count(void* ctx);

/// Dump the loaded tasks as JSON.
DAG_API char* dag_dump_tasks(void* ctx);

// ── Utility ───────────────────────────────────────────────────────────

/// Free a string returned by this DLL.
DAG_API void dag_free_string(char* str);

#ifdef __cplusplus
}
#endif
