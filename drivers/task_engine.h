#pragma once
//
// Minimal task_engine.h for khanary_driver.dll build.
//
// This is a self-contained subset of Kuhul::Runtime::TaskEngine types.
// The full TaskEngine.cpp from .NNC-K/native/runtime/ is NOT linked here;
// khanary_driver.cpp implements its own planning/run logic inline.
//

#include <functional>
#include <string>
#include <vector>

#include "webx_stubs.h"

namespace Kuhul::Runtime {

struct TaskSpec {
    std::string id;
    std::string action;
    std::string description;
    std::string provider;
    std::vector<std::string> dependsOn;
};

struct TaskResult {
    std::string id;
    std::string provider;
    std::string status;
    std::string detail;
};

using TaskExecutor = std::function<bool(
    const TaskSpec&, const WebX::Provider&, std::string& detail)>;

class TaskEngine {
public:
    explicit TaskEngine(WebX::ProviderManager& providers) : providers_(providers) {}

    // Placeholder methods kept for API compatibility with .NNC-K header.
    bool load(const std::string& path, std::string& error) { (void)path; (void)error; return true; }
    bool validate(std::string& error) const { (void)error; return true; }
    std::vector<TaskResult> plan() const { return {}; }
    std::vector<TaskResult> run(const TaskExecutor& executor) const { (void)executor; return {}; }
    const std::vector<TaskSpec>& tasks() const { return tasks_; }

private:
    WebX::ProviderManager& providers_;
    std::vector<TaskSpec> tasks_;
    std::string path_;
};

} // namespace Kuhul::Runtime
