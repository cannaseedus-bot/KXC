#pragma once

#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct DAGTask {
    std::string id;
    std::vector<std::string> dependsOn;
};

struct DAGSchedule {
    std::vector<std::string> ordered;
    std::string error;
};

class DAGScheduler {
public:
    DAGSchedule schedule(const std::vector<DAGTask>& tasks) const;
};

} // namespace Kuhul::Runtime
