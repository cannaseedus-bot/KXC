#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <functional>
#include <algorithm>

using Value = std::variant<std::nullptr_t, int64_t, double, bool, std::string>;
using State = std::unordered_map<std::string, Value>;

enum class XCFEPhase : uint8_t { Pop=0, Wo, Sek, Chen, Xul, Unknown };
XCFEPhase parsePhase(const std::string &s);

struct ExecutionNode {
    XCFEPhase phase = XCFEPhase::Unknown;
    std::string op;
    State state;
    std::vector<ExecutionNode> children;
};

struct ExecutionContext {
    State global;
    State phaseBuffer;
    std::vector<std::string> logs;
    void log(const std::string &m);
};

using OpHandler = std::function<void(ExecutionNode&, ExecutionContext&)>;

class OpRegistry {
public:
    void registerOp(const std::string &name, OpHandler handler);
    bool execute(const std::string &name, ExecutionNode &node, ExecutionContext &ctx) const;
private:
    std::unordered_map<std::string, OpHandler> ops;
};

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {
public:
    ThreadPool(size_t threads = std::thread::hardware_concurrency()) {
        stop = false;
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this]() { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(doneMutex);
        doneCondition.wait(lock, [this]() { return activeTasks == 0 && tasks.empty(); });
    }

    void taskStarted() { activeTasks++; }
    void taskFinished() { if(--activeTasks==0 && tasks.empty()) doneCondition.notify_all(); }

    ~ThreadPool(){
        { std::unique_lock<std::mutex> lock(queueMutex); stop = true; }
        condition.notify_all();
        for (auto &w: workers) if (w.joinable()) w.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop{false};

    std::atomic<int> activeTasks{0};
    std::mutex doneMutex;
    std::condition_variable doneCondition;
};

class ParallelXCFEWalker;

// collect nodes in deterministic preorder (exposed)
void collectPhaseNodes(ExecutionNode &node, XCFEPhase phase, std::vector<ExecutionNode*> &out);

// parse an execution XML file into an ExecutionNode root (simple XML format)
// returns true on success
bool parse_execution_xml(const std::string &path, ExecutionNode &out_root);

// register default core ops (init, assign, add, emit, cleanup)
void registerCoreOps(OpRegistry &reg);

// Execute root in parallel using given threads
auto run_execution(ExecutionNode &root, OpRegistry &reg, size_t threads)->bool;

// New SIMD batching APIs
#include <iostream>
#include <iomanip>
#include "ir.h"

#include "simd_kernels.h"

class BatchExecutor {
public:
    void executePhase(BatchMap &batches, Buffers &buf) {
        // deterministic ordering: iterate over OpCode numeric order
        std::vector<OpCode> order;
        for (auto &p : batches) order.push_back(p.first);
        std::sort(order.begin(), order.end(), [](OpCode a, OpCode b){ return static_cast<uint16_t>(a) < static_cast<uint16_t>(b); });
        for (auto op : order){
            auto &batch = batches[op];
            switch (op){
                case OpCode::ADD_I64:
                    executeAddLane(batch, buf);
                    break;
                case OpCode::BIMODAL_ATTENTION:
                case OpCode::GEODESIC_FLOW:
                    // GPU-targeted ops; CPU path emits a deterministic fallback trace
                    std::cout << "[INFO] BatchExecutor: Executing geometric opcode 0x" 
                              << std::hex << static_cast<int>(op) << std::dec 
                              << " (" << batch.nodes.size() << " nodes)\n";
                    break;
                default:
                    // fallback: scalar execute
                    for (auto *n : batch.nodes){
                        if (n->opcode == OpCode::ADD_I64){
                            buf.i64[n->out_idx] = buf.i64[n->a_idx] + buf.i64[n->b_idx];
                        }
                    }
                    break;
            }
        }
    }
};


// top-level helper to lower ExecutionNode tree to IRNodes, build batches and execute per-phase
void execute_phases_with_batches(ExecutionNode &root, OpRegistry &reg, size_t threads);

