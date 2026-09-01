//
// khanary_driver.cpp — Native driver DLL for TaskEngine + DAG + provider dispatch
//
// Flat C ABI. Compiles with MSVC. Loaded by kuhul-server via ffi-napi/koffi.
//

#define KHANARY_DRIVER_EXPORTS
#include "khanary_driver.h"

#include "webx_stubs.h"
#include "task_engine.h"
#include "dag.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// JSON helpers
// ============================================================================

namespace {

std::string jsonString(const std::string& object, const char* key) {
    const std::string marker = std::string("\"") + key + "\"";
    const size_t pos = object.find(marker);
    if (pos == std::string::npos) return {};
    const size_t colon = object.find(':', pos + marker.size());
    const size_t quote = object.find('"', colon == std::string::npos ? colon : colon + 1);
    if (colon == std::string::npos || quote == std::string::npos) return {};
    const size_t end = object.find('"', quote + 1);
    return end == std::string::npos ? std::string{}
        : object.substr(quote + 1, end - quote - 1);
}

std::vector<std::string> jsonStringArray(const std::string& object, const char* key) {
    const std::string marker = std::string("\"") + key + "\"";
    const size_t pos = object.find(marker);
    if (pos == std::string::npos) return {};
    const size_t open = object.find('[', pos + marker.size());
    const size_t close = object.find(']', open == std::string::npos ? open : open + 1);
    if (open == std::string::npos || close == std::string::npos) return {};

    std::vector<std::string> values;
    size_t cursor = open + 1;
    while (cursor < close) {
        const size_t q1 = object.find('"', cursor);
        if (q1 == std::string::npos || q1 >= close) break;
        const size_t q2 = object.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 > close) break;
        values.push_back(object.substr(q1 + 1, q2 - q1 - 1));
        cursor = q2 + 1;
    }
    return values;
}

bool loadJsonTasks(const std::string& source,
                   std::vector<Kuhul::Runtime::TaskSpec>& tasks,
                   std::string& error) {
    const size_t tasksKey = source.find("\"tasks\"");
    if (tasksKey == std::string::npos) {
        error = "json_tasks_array_missing";
        return false;
    }
    const size_t arrayStart = source.find('[', tasksKey);
    if (arrayStart == std::string::npos) {
        error = "json_tasks_array_invalid";
        return false;
    }
    size_t arrayEnd = std::string::npos;
    size_t depth = 0;
    bool quoted = false;
    for (size_t i = arrayStart; i < source.size(); ++i) {
        if (source[i] == '"' && (i == 0 || source[i - 1] != '\\'))
            quoted = !quoted;
        if (quoted) continue;
        if (source[i] == '[') ++depth;
        if (source[i] == ']' && --depth == 0) { arrayEnd = i; break; }
    }
    if (arrayEnd == std::string::npos) {
        error = "json_tasks_array_invalid";
        return false;
    }

    size_t cursor = arrayStart + 1;
    while (cursor < arrayEnd) {
        const size_t open = source.find('{', cursor);
        if (open == std::string::npos || open >= arrayEnd) break;
        const size_t close = source.find('}', open + 1);
        if (close == std::string::npos || close > arrayEnd) {
            error = "json_task_object_invalid";
            return false;
        }
        const std::string object = source.substr(open, close - open + 1);
        Kuhul::Runtime::TaskSpec task;
        task.id = jsonString(object, "id");
        task.action = jsonString(object, "action");
        task.description = jsonString(object, "description");
        task.provider = jsonString(object, "provider");
        task.dependsOn = jsonStringArray(object, "depends_on");
        if (task.dependsOn.empty())
            task.dependsOn = jsonStringArray(object, "depends");
        if (task.id.empty()) {
            error = "json_task_missing_id";
            return false;
        }
        tasks.push_back(std::move(task));
        cursor = close + 1;
    }
    return true;
}

std::vector<WebX::Provider> parseProviders(const std::string& json) {
    std::vector<WebX::Provider> providers;
    size_t cursor = 0;
    while (true) {
        const size_t open = json.find('{', cursor);
        if (open == std::string::npos) break;
        const size_t close = json.find('}', open + 1);
        if (close == std::string::npos) break;
        const std::string obj = json.substr(open, close - open + 1);
        WebX::Provider p;
        p.id = jsonString(obj, "id");
        p.library = jsonString(obj, "library");
        const std::string avail = jsonString(obj, "available");
        p.available = avail.empty() || avail == "true";
        if (!p.id.empty()) providers.push_back(p);
        cursor = close + 1;
    }
    return providers;
}

std::string resultsToJson(const std::vector<Kuhul::Runtime::TaskResult>& results) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) out << ",";
        const auto& r = results[i];
        out << "{\"id\":\"" << r.id
            << "\",\"provider\":\"" << r.provider
            << "\",\"status\":\"" << r.status
            << "\",\"detail\":\"" << r.detail << "\"}";
    }
    out << "]";
    return out.str();
}

// ============================================================================
// Driver engine — same core as task_engine_wasm.cpp, kd_ prefix
// ============================================================================

struct DriverEngine {
    std::unique_ptr<WebX::ProviderManager> providers;
    std::unique_ptr<Kuhul::Runtime::TaskEngine> engine;
    std::vector<Kuhul::Runtime::TaskSpec> loadedTasks;
    std::string lastError;

    DriverEngine(const std::vector<WebX::Provider>& providerList)
        : providers(new WebX::ProviderManager(providerList))
        , engine(new Kuhul::Runtime::TaskEngine(*providers))
    {}

    bool loadJson(const std::string& json) {
        loadedTasks.clear();
        lastError.clear();
        if (!loadJsonTasks(json, loadedTasks, lastError)) return false;
        return validateInline();
    }

    bool validateInline() {
        std::set<std::string> ids;
        for (const auto& task : loadedTasks) {
            if (task.id.empty() || !ids.insert(task.id).second) {
                lastError = "task_ids_must_be_unique";
                return false;
            }
        }
        for (const auto& task : loadedTasks) {
            for (const auto& dep : task.dependsOn) {
                if (ids.find(dep) == ids.end()) {
                    lastError = "missing_task_dependency:" + dep;
                    return false;
                }
            }
        }
        if (loadedTasks.empty()) {
            lastError = "task_list_is_empty";
            return false;
        }
        return true;
    }

    std::vector<Kuhul::Runtime::TaskResult> plan() {
        std::vector<Kuhul::Runtime::TaskResult> results;
        std::set<std::string> completed, failed, remaining;
        for (const auto& task : loadedTasks) remaining.insert(task.id);

        while (!remaining.empty()) {
            bool progressed = false;
            for (const auto& task : loadedTasks) {
                if (!remaining.count(task.id)) continue;

                if (std::any_of(task.dependsOn.begin(), task.dependsOn.end(),
                                [&failed](const std::string& id) {
                                    return failed.count(id) != 0;
                                })) {
                    results.push_back({task.id, task.provider, "blocked", "dependency_failed"});
                    failed.insert(task.id);
                    completed.insert(task.id);
                    remaining.erase(task.id);
                    progressed = true;
                    continue;
                }

                if (!std::all_of(task.dependsOn.begin(), task.dependsOn.end(),
                                 [&completed](const std::string& id) {
                                     return completed.count(id) != 0;
                                 }))
                    continue;

                const auto& providerList = providers->getProviders();
                const auto prov = std::find_if(
                    providerList.begin(), providerList.end(),
                    [&task](const auto& p) { return p.id == task.provider; });

                if (prov == providerList.end()) {
                    results.push_back({task.id, task.provider, "blocked", "provider_not_registered"});
                    failed.insert(task.id);
                } else if (!prov->available) {
                    results.push_back({task.id, prov->id, "blocked", "provider_not_available"});
                    failed.insert(task.id);
                } else {
                    results.push_back({task.id, prov->id, "admitted", "driver_provider_ready"});
                }
                completed.insert(task.id);
                remaining.erase(task.id);
                progressed = true;
            }
            if (!progressed) {
                for (const auto& id : remaining)
                    results.push_back({id, {}, "blocked", "dependency_cycle"});
                break;
            }
        }
        return results;
    }

    std::vector<Kuhul::Runtime::TaskResult> run() {
        const auto scheduled = plan();
        std::set<std::string> completed, failed;
        std::vector<Kuhul::Runtime::TaskResult> results;

        for (const auto& r : scheduled) {
            if (r.status == "blocked") { failed.insert(r.id); }
        }

        for (const auto& r : scheduled) {
            if (r.status != "admitted") {
                results.push_back(r);
                continue;
            }
            const auto task = std::find_if(
                loadedTasks.begin(), loadedTasks.end(),
                [&r](const auto& t) { return t.id == r.id; });
            bool depsFailed = false;
            if (task != loadedTasks.end()) {
                for (const auto& dep : task->dependsOn) {
                    if (failed.count(dep)) { depsFailed = true; break; }
                }
            }
            if (depsFailed) {
                results.push_back({r.id, r.provider, "blocked", "dependency_failed"});
                failed.insert(r.id);
            } else {
                results.push_back({r.id, r.provider, "planned", "ready_for_mcp_dispatch"});
                completed.insert(r.id);
            }
        }
        return results;
    }

    Kuhul::Runtime::TaskResult dispatchOne(const std::string& taskJson) {
        Kuhul::Runtime::TaskResult result;
        result.id = jsonString(taskJson, "id");
        result.provider = jsonString(taskJson, "provider");
        if (result.id.empty()) {
            result.status = "failed";
            result.detail = "dispatch_missing_task_id";
            return result;
        }

        // Validate task exists in loaded list
        const auto task = std::find_if(
            loadedTasks.begin(), loadedTasks.end(),
            [&result](const auto& t) { return t.id == result.id; });
        if (task == loadedTasks.end()) {
            result.status = "failed";
            result.detail = "dispatch_task_not_found:" + result.id;
            return result;
        }

        // Check provider
        const auto& providerList = providers->getProviders();
        const auto prov = std::find_if(
            providerList.begin(), providerList.end(),
            [&result](const auto& p) { return p.id == result.provider; });
        if (prov == providerList.end()) {
            result.status = "blocked";
            result.detail = "provider_not_registered";
            return result;
        }
        if (!prov->available) {
            result.status = "blocked";
            result.detail = "provider_not_available";
            return result;
        }

        // Admitted — ready for MCP execution on JS side
        result.status = "dispatched";
        result.detail = "sent_to_mcp:" + task->action;
        return result;
    }

    int taskCount() const { return static_cast<int>(loadedTasks.size()); }
};

} // namespace

// ============================================================================
// C ABI exports
// ============================================================================

extern "C" {

KHANARY_API void* kd_create(const char* providers_json) {
    if (!providers_json) return nullptr;
    auto providers = parseProviders(providers_json);
    if (providers.empty()) {
        providers.push_back({"wasm", "task_engine.wasm", true});
    }
    return new DriverEngine(providers);
}

KHANARY_API void kd_destroy(void* driver) {
    delete static_cast<DriverEngine*>(driver);
}

KHANARY_API int kd_load_tasks(void* driver, const char* tasklist_json,
                               char* error_buf, int error_buf_size) {
    if (!driver || !tasklist_json) {
        if (error_buf && error_buf_size > 0) {
            std::strncpy(error_buf, "null_driver_or_input", error_buf_size - 1);
        }
        return 0;
    }
    auto* d = static_cast<DriverEngine*>(driver);
    const bool ok = d->loadJson(tasklist_json);
    if (!ok && error_buf && error_buf_size > 0) {
        std::strncpy(error_buf, d->lastError.c_str(), error_buf_size - 1);
    }
    return ok ? 1 : 0;
}

KHANARY_API char* kd_plan(void* driver) {
    if (!driver) {
        auto* empty = new char[3];
        std::strcpy(empty, "[]");
        return empty;
    }
    auto* d = static_cast<DriverEngine*>(driver);
    const auto results = d->plan();
    const std::string json = resultsToJson(results);
    auto* out = new char[json.size() + 1];
    std::memcpy(out, json.data(), json.size());
    out[json.size()] = '\0';
    return out;
}

KHANARY_API char* kd_run(void* driver) {
    if (!driver) {
        auto* empty = new char[3];
        std::strcpy(empty, "[]");
        return empty;
    }
    auto* d = static_cast<DriverEngine*>(driver);
    const auto results = d->run();
    const std::string json = resultsToJson(results);
    auto* out = new char[json.size() + 1];
    std::memcpy(out, json.data(), json.size());
    out[json.size()] = '\0';
    return out;
}

KHANARY_API char* kd_dispatch(void* driver, const char* task_json) {
    if (!driver || !task_json) {
        const char* err = "{\"status\":\"failed\",\"detail\":\"null_driver_or_input\"}";
        auto* out = new char[std::strlen(err) + 1];
        std::strcpy(out, err);
        return out;
    }
    auto* d = static_cast<DriverEngine*>(driver);
    const auto result = d->dispatchOne(task_json);
    const std::string json = "{\"id\":\"" + result.id +
        "\",\"provider\":\"" + result.provider +
        "\",\"status\":\"" + result.status +
        "\",\"detail\":\"" + result.detail + "\"}";
    auto* out = new char[json.size() + 1];
    std::memcpy(out, json.data(), json.size());
    out[json.size()] = '\0';
    return out;
}

KHANARY_API int kd_task_count(void* driver) {
    if (!driver) return 0;
    return static_cast<DriverEngine*>(driver)->taskCount();
}

KHANARY_API void kd_free_string(char* str) {
    delete[] str;
}

} // extern "C"
