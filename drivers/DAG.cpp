#include "dag.h"

#include <algorithm>
#include <map>
#include <set>

namespace Kuhul::Runtime {

DAGSchedule DAGScheduler::schedule(
    const std::vector<DAGTask>& tasks) const {
    DAGSchedule result;
    std::map<std::string, DAGTask> byId;
    std::map<std::string, std::size_t> indegree;
    std::map<std::string, std::vector<std::string>> dependents;

    for (const auto& task : tasks) {
        if (task.id.empty() || !byId.emplace(task.id, task).second) {
            result.error = "dag_task_ids_must_be_unique";
            return result;
        }
        indegree[task.id] = task.dependsOn.size();
    }
    for (const auto& task : tasks) {
        for (const auto& dependency : task.dependsOn) {
            if (!byId.count(dependency)) {
                result.error = "dag_missing_dependency:" + dependency;
                return result;
            }
            dependents[dependency].push_back(task.id);
        }
    }

    std::set<std::string> ready;
    for (const auto& entry : indegree)
        if (entry.second == 0) ready.insert(entry.first);

    while (!ready.empty()) {
        const std::string id = *ready.begin();
        ready.erase(ready.begin());
        result.ordered.push_back(id);
        for (const auto& dependent : dependents[id]) {
            if (--indegree[dependent] == 0) ready.insert(dependent);
        }
    }
    if (result.ordered.size() != tasks.size()) {
        result.ordered.clear();
        result.error = "dag_cycle_detected";
    }
    return result;
}

} // namespace Kuhul::Runtime
