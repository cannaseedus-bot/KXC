// kuhul_tool_runtime.h — JSONL-based K'UHUL tool execution runtime
//
// Loads tool definitions from kuhul.tools.jsonl. Each line is a complete,
// independent tool definition — batchable, parallelisable, sandboxable.
//
// JSONL format per line:
//   {"id":"tool_001","type":"function","name":"read_file","version":"1.0.0",
//    "effect":"io","permissions":["fs.read"],"sandbox":"restricted",
//    "batchable":true,"thread_safe":true,"timeout_ms":5000,
//    "cmd":"kuhul.fs.read","args":["path:string"],"returns":"string",
//    "help":"Read entire file contents"}
//
// Types: function | agent | micronaut | batch | shell | http | agent_route | sandbox | help
// Sandboxes: restricted | isolated | gpu_sandbox | thread_pool |
//            process_isolation | network_sandbox | api_gateway |
//            full_isolation | readonly
//
// K'UHUL integration:
//   call_tool("read_file", ["data.txt"])
//   batch_call_tools([{"tool":"read_file","args":["f1.txt"]}, ...])
//   call_agent_route("/api/agent/query", "POST", headers, body)

#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace Kuhul {
namespace Tools {

// ─── ToolDefinition ───────────────────────────────────────────────────────────

struct ToolDefinition {
    std::string              id, type, name, version, effect;
    std::vector<std::string> permissions;
    std::string              sandbox, cmd, returns, aggregator, capsule, help;
    bool                     batchable     = false;
    bool                     thread_safe   = false;
    bool                     api_key_req   = false;
    int                      timeout_ms    = 5000;
    int                      max_batch     = 1;
    int                      rate_limit    = 0;
    std::string              route, method;
    std::vector<std::string> args, sub_tools;
    std::map<std::string, int>                               resources;
    std::vector<std::map<std::string, std::string>>          routes;

    // Minimal JSON-line parser (no external deps)
    static std::string ExtractStr(const std::string& line, const std::string& key) {
        std::string k = "\"" + key + "\"";
        auto p = line.find(k);
        if (p == std::string::npos) return {};
        auto q = line.find('"', p + k.size() + 1);
        if (q == std::string::npos) return {};
        auto r = line.find('"', q + 1);
        return r == std::string::npos ? std::string{} : line.substr(q + 1, r - q - 1);
    }
    static bool ExtractBool(const std::string& line, const std::string& key) {
        std::string k = "\"" + key + "\":";
        auto p = line.find(k);
        if (p == std::string::npos) return false;
        auto v = line.find_first_not_of(" \t", p + k.size());
        return v != std::string::npos && line[v] == 't';
    }
    static int ExtractInt(const std::string& line, const std::string& key) {
        std::string k = "\"" + key + "\":";
        auto p = line.find(k);
        if (p == std::string::npos) return 0;
        auto v = line.find_first_of("0123456789-", p + k.size());
        return v == std::string::npos ? 0 : std::stoi(line.substr(v));
    }

    static ToolDefinition Parse(const std::string& line) {
        ToolDefinition t;
        t.id          = ExtractStr(line, "id");
        t.type        = ExtractStr(line, "type");
        t.name        = ExtractStr(line, "name");
        t.version     = ExtractStr(line, "version");
        t.effect      = ExtractStr(line, "effect");
        t.sandbox     = ExtractStr(line, "sandbox");
        t.cmd         = ExtractStr(line, "cmd");
        t.returns     = ExtractStr(line, "returns");
        t.aggregator  = ExtractStr(line, "aggregator");
        t.capsule     = ExtractStr(line, "capsule");
        t.help        = ExtractStr(line, "help");
        t.route       = ExtractStr(line, "route");
        t.method      = ExtractStr(line, "method");
        t.batchable   = ExtractBool(line, "batchable");
        t.thread_safe = ExtractBool(line, "thread_safe");
        t.api_key_req = ExtractBool(line, "api_key_required");
        t.timeout_ms  = ExtractInt(line, "timeout_ms");
        t.max_batch   = ExtractInt(line, "max_batch_size");
        t.rate_limit  = ExtractInt(line, "rate_limit");
        return t;
    }
};

// ─── ToolRegistry ─────────────────────────────────────────────────────────────

class ToolRegistry {
    std::map<std::string, ToolDefinition>           by_id_;
    std::map<std::string, std::vector<std::string>> by_type_;
    mutable std::mutex                              mu_;
public:
    void LoadFromJSONL(const std::string& path) {
        std::ifstream f(path);
        std::string   line;
        std::lock_guard<std::mutex> lg(mu_);
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto t = ToolDefinition::Parse(line);
            if (!t.id.empty()) {
                by_type_[t.type].push_back(t.id);
                by_id_[t.id] = std::move(t);
            }
        }
    }

    const ToolDefinition* Get(const std::string& id) const {
        std::lock_guard<std::mutex> lg(mu_);
        auto it = by_id_.find(id);
        return it != by_id_.end() ? &it->second : nullptr;
    }

    std::vector<const ToolDefinition*> ByType(const std::string& type) const {
        std::lock_guard<std::mutex> lg(mu_);
        std::vector<const ToolDefinition*> out;
        auto it = by_type_.find(type);
        if (it != by_type_.end())
            for (const auto& id : it->second)
                if (auto* t = Get(id)) out.push_back(t);
        return out;
    }

    size_t Count() const {
        std::lock_guard<std::mutex> lg(mu_);
        return by_id_.size();
    }
};

// ─── BatchExecutor ────────────────────────────────────────────────────────────

class BatchExecutor {
    std::vector<std::thread>         pool_;
    std::queue<std::function<void()>> q_;
    std::mutex                        mu_;
    std::condition_variable           cv_;
    bool                              stop_ = false;
public:
    explicit BatchExecutor(size_t n = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < n; i++)
            pool_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    { std::unique_lock<std::mutex> lk(mu_);
                      cv_.wait(lk, [this]{ return stop_ || !q_.empty(); });
                      if (stop_ && q_.empty()) return;
                      task = std::move(q_.front()); q_.pop(); }
                    task();
                }
            });
    }
    ~BatchExecutor() {
        { std::unique_lock<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& w : pool_) w.join();
    }

    template<class F>
    auto Enqueue(F&& f) -> std::future<decltype(f())> {
        auto p = std::make_shared<std::packaged_task<decltype(f())()>>(std::forward<F>(f));
        auto fut = p->get_future();
        { std::lock_guard<std::mutex> lk(mu_); q_.emplace([p]{ (*p)(); }); }
        cv_.notify_one();
        return fut;
    }

    std::vector<std::string> RunBatch(
        const std::vector<std::function<std::string()>>& tasks)
    {
        std::vector<std::future<std::string>> futs;
        futs.reserve(tasks.size());
        for (const auto& t : tasks) futs.push_back(Enqueue(t));
        std::vector<std::string> res;
        res.reserve(futs.size());
        for (auto& fut : futs) res.push_back(fut.get());
        return res;
    }
};

// ─── SandboxPolicy ────────────────────────────────────────────────────────────

class SandboxPolicy {
    std::map<std::string, std::function<bool(const std::string&)>> policies_;
public:
    SandboxPolicy() {
        policies_["restricted"]        = [](const std::string& op){ return op.find("fs.read")==0 || op.find("help")==0; };
        policies_["isolated"]          = [](const std::string& op){ return op.find("sandbox")==0||op.find("compute")==0||op.find("ai")==0; };
        policies_["gpu_sandbox"]       = [](const std::string& op){ return op.find("compute.gpu")==0||op.find("memory")==0||op.find("ai.infer")==0; };
        policies_["thread_pool"]       = [](const std::string& op){ return op.find("compute")==0; };
        policies_["process_isolation"] = [](const std::string& op){ return op.find("os.exec")==0||op.find("os.proc")==0; };
        policies_["network_sandbox"]   = [](const std::string& op){ return op.find("network")==0; };
        policies_["api_gateway"]       = [](const std::string& op){ return op.find("api")==0; };
        policies_["full_isolation"]    = [](const std::string&     ){ return false; };
        policies_["readonly"]          = [](const std::string& op){ return op.find("read")!=std::string::npos||op.find("help")!=std::string::npos; };
    }

    bool Allow(const std::string& sandbox, const std::string& permission) const {
        auto it = policies_.find(sandbox);
        return it != policies_.end() && it->second(permission);
    }
};

// ─── ToolExecutor ─────────────────────────────────────────────────────────────

class ToolExecutor {
    ToolRegistry   registry_;
    BatchExecutor  batch_;
    SandboxPolicy  sandbox_;
    std::map<std::string, int>                             rate_counters_;
    std::map<std::string, std::chrono::steady_clock::time_point> rate_reset_;
    std::mutex                                             rate_mu_;

    bool RateCheck(const std::string& key, int limit) {
        if (limit <= 0) return true;
        std::lock_guard<std::mutex> lk(rate_mu_);
        auto now = std::chrono::steady_clock::now();
        auto& reset = rate_reset_[key];
        if (now > reset) { rate_counters_[key] = 1; reset = now + std::chrono::seconds(60); return true; }
        return ++rate_counters_[key] <= limit;
    }

public:
    void LoadTools(const std::string& jsonl_path) {
        registry_.LoadFromJSONL(jsonl_path);
    }
    size_t ToolCount() const { return registry_.Count(); }

    std::string Call(const std::string& tool_id, const std::vector<std::string>& args) {
        const auto* tool = registry_.Get(tool_id);
        if (!tool) return "Error: tool not found: " + tool_id;
        if (!RateCheck(tool_id, tool->rate_limit)) return "Error: rate limit exceeded";
        for (const auto& perm : tool->permissions)
            if (!sandbox_.Allow(tool->sandbox, perm))
                return "Error: permission denied (" + perm + ") in sandbox=" + tool->sandbox;

        if (tool->type == "function" || tool->type == "shell") return ExecFunction(*tool, args);
        if (tool->type == "agent")     return "[" + tool->name + "] " + (args.empty() ? "" : args[0]);
        if (tool->type == "micronaut") return "Spawned " + tool->capsule + " (id=" + std::to_string(std::hash<std::string>{}(tool->id)) + ")";
        if (tool->type == "http")      return "HTTP " + (args.size()>0?args[0]:"GET") + " " + (args.size()>1?args[1]:"");
        if (tool->type == "help")      return GetHelp(args.empty() ? "" : args[0]);
        if (tool->type == "batch")     return ExecBatch(*tool, args);
        return "Executed: " + tool->name;
    }

    std::vector<std::string> BatchCall(
        const std::vector<std::pair<std::string, std::vector<std::string>>>& calls)
    {
        std::vector<std::function<std::string()>> tasks;
        tasks.reserve(calls.size());
        for (const auto& c : calls)
            tasks.push_back([this, c]{ return Call(c.first, c.second); });
        return batch_.RunBatch(tasks);
    }

    std::string GetHelp(const std::string& tool_id) const {
        const auto* t = registry_.Get(tool_id);
        if (!t) return "Tool not found: " + tool_id;
        std::ostringstream s;
        s << t->name << " v" << t->version << "\n"
          << "  type=" << t->type << "  effect=" << t->effect << "  sandbox=" << t->sandbox << "\n"
          << "  batchable=" << (t->batchable?"yes":"no") << "  thread_safe=" << (t->thread_safe?"yes":"no") << "\n"
          << "  timeout=" << t->timeout_ms << "ms  rate_limit=" << t->rate_limit << "/min\n"
          << "  " << t->help;
        return s.str();
    }

private:
    std::string ExecFunction(const ToolDefinition& t, const std::vector<std::string>& args) {
        if (t.cmd == "kuhul.fs.read") {
            if (args.empty()) return "Error: path required";
            std::ifstream f(args[0]);
            if (!f) return "Error: cannot open: " + args[0];
            return std::string(std::istreambuf_iterator<char>(f), {});
        }
        if (t.cmd == "kuhul.fs.write") {
            if (args.size() < 2) return "Error: path+content required";
            std::ofstream f(args[0]);
            if (!f) return "false";
            f << args[1]; return "true";
        }
        if (t.cmd == "os.exec") {
            if (args.empty()) return "Error: cmd required";
            std::string cmd = args[0];
            for (size_t i=1; i<args.size(); i++) cmd += " " + args[i];
            char buf[128]; std::string out;
#ifdef _WIN32
            std::unique_ptr<FILE,decltype(&_pclose)> p(_popen(cmd.c_str(),"r"),_pclose);
#else
            std::unique_ptr<FILE,decltype(&pclose)>  p(popen(cmd.c_str(),"r"), pclose);
#endif
            if (!p) return "Error: cannot exec";
            while (fgets(buf, sizeof buf, p.get())) out += buf;
            return out;
        }
        return "Executed: " + t.cmd;
    }

    std::string ExecBatch(const ToolDefinition& t, const std::vector<std::string>& args) {
        std::vector<std::pair<std::string,std::vector<std::string>>> calls;
        for (const auto& sub : t.sub_tools) calls.emplace_back(sub, args);
        auto results = BatchCall(calls);
        if (t.aggregator == "concat") {
            std::string out;
            for (const auto& r : results) out += r + "\n";
            return out;
        }
        return results.empty() ? "" : results[0];
    }
};

// ─── Global instance ──────────────────────────────────────────────────────────

inline ToolExecutor& GlobalExecutor() {
    static ToolExecutor inst;
    return inst;
}

inline void LoadTools(const std::string& jsonl_path) {
    GlobalExecutor().LoadTools(jsonl_path);
}

inline std::string CallTool(const std::string& id, const std::vector<std::string>& args) {
    return GlobalExecutor().Call(id, args);
}

inline std::vector<std::string> BatchCallTools(
    const std::vector<std::pair<std::string,std::vector<std::string>>>& calls)
{
    return GlobalExecutor().BatchCall(calls);
}

} // namespace Tools
} // namespace Kuhul
