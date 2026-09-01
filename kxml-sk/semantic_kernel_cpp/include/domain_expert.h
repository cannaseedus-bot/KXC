// ============================================================================
// domain_expert.h - Specialized Expertise for Micronauts (ASX v0.7)
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace asx {

enum class ExpertSpecialization {
    CODING,
    FACTORY,
    REASONING,
    SECURITY
};

struct ExpertResponse {
    std::string task;
    float confidence;
    std::string output;
    std::map<std::string, float> manifold_deltas; // Impact on tensor fields
};

class DomainExpert {
public:
    virtual ~DomainExpert() = default;
    virtual ExpertSpecialization get_specialization() const = 0;
    virtual ExpertResponse execute_task(const std::string& task, const std::string& input) = 0;
};

// ⟁ AgentCoder: Domain Expert for Software Engineering
class AgentCoder : public DomainExpert {
public:
    ExpertSpecialization get_specialization() const override { return ExpertSpecialization::CODING; }
    ExpertResponse execute_task(const std::string& task, const std::string& input) override {
        ExpertResponse res;
        res.task = task;
        res.confidence = 0.95f;
        // In a real implementation, this calls CodeReviewEngine/DolphinCoder
        res.output = "[AgentCoder] Executing " + task + " on provided code.";
        res.manifold_deltas["--attention"] = 0.1f; // Coding requires high focus
        res.manifold_deltas["--entropy"] = -0.05f; // Coding reduces entropy (structures data)
        return res;
    }
};

// ⟁ AgentFactory: Domain Expert for Micronaut Instantiation
class AgentFactory : public DomainExpert {
public:
    ExpertSpecialization get_specialization() const override { return ExpertSpecialization::FACTORY; }
    ExpertResponse execute_task(const std::string& task, const std::string& input) override {
        ExpertResponse res;
        res.task = task;
        res.confidence = 0.88f;
        // In a real implementation, this calls MicronauntFactory
        res.output = "[AgentFactory] Manufacturing new micronaut for domain: " + input;
        res.manifold_deltas["--pressure"] = 0.15f; // Creation increases system pressure
        res.manifold_deltas["--entropy"] = 0.1f;  // New domains increase exploration pressure
        return res;
    }
};

} // namespace asx
