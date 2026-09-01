//
// dag_abi.cpp — C ABI wrapper around Kuhul::Runtime::DAGScheduler
//

#define DAG_EXPORTS
#include "dag_abi.h"
#include "dag.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

// Simple in-memory JSON helper: we avoid external dependencies so the DLL
// is self-contained and can be dropped next to kxc.exe / kuhul-server.
namespace {

std::string json_escape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    out += '"';
    return out;
}

std::string json_array(const std::vector<std::string>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) out += ",";
        out += json_escape(items[i]);
    }
    out += "]";
    return out;
}

std::string make_schedule_result(const std::vector<std::string>& ordered,
                                 const std::string& error) {
    return "{\"ordered\":" + json_array(ordered) +
           ",\"error\":" + json_escape(error) + "}";
}

std::string make_tasks_dump(const std::vector<Kuhul::Runtime::DAGTask>& tasks) {
    std::string out = "[";
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (i > 0) out += ",";
        out += "{\"id\":" + json_escape(tasks[i].id) +
               ",\"dependsOn\":" + json_array(tasks[i].dependsOn) + "}";
    }
    out += "]";
    return out;
}

// Minimal JSON parser for task arrays.
// Supports: [{"id":"a","depends_on":["b"],"dependsOn":["c"]}]
bool parse_task_array(const std::string& json,
                      std::vector<Kuhul::Runtime::DAGTask>& out_tasks,
                      std::string& error) {
    out_tasks.clear();
    error.clear();

    size_t i = 0;
    const size_t n = json.size();

    auto skip_ws = [&]() {
        while (i < n && (json[i] == ' ' || json[i] == '\t' ||
                         json[i] == '\n' || json[i] == '\r')) {
            ++i;
        }
    };

    auto expect = [&](char c) -> bool {
        skip_ws();
        if (i < n && json[i] == c) { ++i; return true; }
        return false;
    };

    auto parse_string = [&](std::string& out) -> bool {
        skip_ws();
        if (i >= n || json[i] != '"') return false;
        ++i;
        out.clear();
        while (i < n) {
            char c = json[i++];
            if (c == '"') return true;
            if (c == '\\' && i < n) {
                char esc = json[i++];
                switch (esc) {
                    case '"': case '\\': out += esc; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    default: out += esc; break;
                }
            } else {
                out += c;
            }
        }
        return false;
    };

    if (!expect('[')) {
        error = "expected '[' at start of task array";
        return false;
    }
    skip_ws();
    if (i < n && json[i] == ']') { ++i; return true; }

    while (i < n) {
        if (!expect('{')) {
            error = "expected '{' for task object";
            return false;
        }

        Kuhul::Runtime::DAGTask task;
        std::string key;
        bool has_id = false;

        while (true) {
            skip_ws();
            if (i < n && json[i] == '}') { ++i; break; }
            if (!parse_string(key)) {
                error = "expected task object key";
                return false;
            }
            if (!expect(':')) {
                error = "expected ':' after key '" + key + "'";
                return false;
            }
            if (key == "id") {
                if (!parse_string(task.id)) {
                    error = "expected string value for 'id'";
                    return false;
                }
                has_id = true;
            } else if (key == "dependsOn" || key == "depends_on") {
                skip_ws();
                if (i >= n || json[i] != '[') {
                    error = "expected '[' for dependency array";
                    return false;
                }
                ++i;
                skip_ws();
                if (i < n && json[i] == ']') { ++i; }
                else {
                    while (i < n) {
                        std::string dep;
                        if (!parse_string(dep)) {
                            error = "expected dependency string";
                            return false;
                        }
                        task.dependsOn.push_back(dep);
                        skip_ws();
                        if (i < n && json[i] == ',') { ++i; continue; }
                        if (i < n && json[i] == ']') { ++i; break; }
                        error = "expected ',' or ']' in dependency array";
                        return false;
                    }
                }
            } else {
                // Unknown key: skip primitive value (best-effort).
                skip_ws();
                if (i < n && json[i] == '"') {
                    std::string tmp;
                    parse_string(tmp);
                } else if (i < n && json[i] == '[') {
                    int depth = 1;
                    ++i;
                    while (i < n && depth > 0) {
                        if (json[i] == '[') ++depth;
                        else if (json[i] == ']') --depth;
                        ++i;
                    }
                } else if (i < n && json[i] == '{') {
                    int depth = 1;
                    ++i;
                    while (i < n && depth > 0) {
                        if (json[i] == '{') ++depth;
                        else if (json[i] == '}') --depth;
                        ++i;
                    }
                } else {
                    while (i < n && json[i] != ',' && json[i] != '}') ++i;
                }
            }
            skip_ws();
            if (i < n && json[i] == ',') { ++i; continue; }
            if (i < n && json[i] == '}') { ++i; break; }
        }

        if (!has_id || task.id.empty()) {
            error = "task missing required 'id' field";
            return false;
        }
        out_tasks.push_back(std::move(task));

        skip_ws();
        if (i < n && json[i] == ',') { ++i; continue; }
        if (i < n && json[i] == ']') { ++i; return true; }
    }

    error = "unexpected end of task array";
    return false;
}

} // namespace

struct DagContext {
    std::vector<Kuhul::Runtime::DAGTask> tasks;
    std::string last_error;
};

extern "C" {

DAG_API void* dag_create(void) {
    return new DagContext();
}

DAG_API void dag_destroy(void* ctx) {
    delete static_cast<DagContext*>(ctx);
}

DAG_API int dag_load_tasks(void* ctx, const char* tasks_json,
                           char* error_buf, int error_buf_size) {
    if (!ctx) return 0;
    auto* dc = static_cast<DagContext*>(ctx);
    dc->tasks.clear();
    dc->last_error.clear();

    std::string error;
    if (!parse_task_array(tasks_json ? tasks_json : "", dc->tasks, error)) {
        dc->last_error = error;
        if (error_buf && error_buf_size > 0) {
            std::strncpy(error_buf, error.c_str(), error_buf_size - 1);
            error_buf[error_buf_size - 1] = '\0';
        }
        return 0;
    }
    return 1;
}

DAG_API char* dag_schedule(void* ctx) {
    if (!ctx) {
        return _strdup(make_schedule_result({}, "null_context").c_str());
    }
    auto* dc = static_cast<DagContext*>(ctx);
    Kuhul::Runtime::DAGScheduler scheduler;
    auto result = scheduler.schedule(dc->tasks);

    std::string out = make_schedule_result(result.ordered, result.error);
    return _strdup(out.c_str());
}

DAG_API char* dag_schedule_json(const char* tasks_json,
                                char* error_buf, int error_buf_size) {
    void* ctx = dag_create();
    if (!ctx) {
        if (error_buf && error_buf_size > 0) {
            std::strncpy(error_buf, "failed to create DAG context", error_buf_size - 1);
            error_buf[error_buf_size - 1] = '\0';
        }
        return _strdup(make_schedule_result({}, "create_failed").c_str());
    }

    int loaded = dag_load_tasks(ctx, tasks_json, error_buf, error_buf_size);
    char* result = nullptr;
    if (loaded) {
        result = dag_schedule(ctx);
    } else {
        std::string err = "load_failed";
        if (error_buf && error_buf_size > 0) err = error_buf;
        result = _strdup(make_schedule_result({}, err).c_str());
    }
    dag_destroy(ctx);
    return result;
}

DAG_API int dag_task_count(void* ctx) {
    if (!ctx) return 0;
    return static_cast<int>(static_cast<DagContext*>(ctx)->tasks.size());
}

DAG_API char* dag_dump_tasks(void* ctx) {
    if (!ctx) return _strdup("[]");
    auto* dc = static_cast<DagContext*>(ctx);
    return _strdup(make_tasks_dump(dc->tasks).c_str());
}

DAG_API void dag_free_string(char* str) {
    if (str) std::free(str);
}

} // extern "C"
