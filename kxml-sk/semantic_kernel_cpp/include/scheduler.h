#pragma once
#include <string>
#include <mutex>
#include <unordered_map>

struct StageState {
    std::string id;
    int credits = 0;
    int max_credits = 4;
};

class CreditScheduler {
public:
    CreditScheduler(){}
    void init_stage(const std::string &id, int max_credits){
        std::lock_guard<std::mutex> lk(mutex_);
        StageState s; s.id = id; s.max_credits = max_credits; s.credits = max_credits;
        stages_[id] = s;
    }

    // try to acquire a credit for stage; returns true and decrements credit
    bool try_acquire(const std::string &id){
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = stages_.find(id);
        if (it==stages_.end()) return false;
        if (it->second.credits <= 0) return false;
        it->second.credits -= 1;
        return true;
    }

    // release credit back to stage
    void release(const std::string &id){
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = stages_.find(id);
        if (it==stages_.end()) return;
        it->second.credits = std::min(it->second.credits + 1, it->second.max_credits);
    }

    int get_credits(const std::string &id){ std::lock_guard<std::mutex> lk(mutex_); auto it=stages_.find(id); if (it==stages_.end()) return 0; return it->second.credits; }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, StageState> stages_;
};
