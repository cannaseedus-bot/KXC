// kuhul_functions.h — Native function bindings for K'UHUL
//
// The core realization:
//   Compute = pi-field is pure. No side effects.
//   Action  = functions have side effects. They touch the world.
//   Bridge  = functions are just external nodes with bindings.
//   Rule    = KUHUL calls functions. Functions don't call KUHUL.
//
// Function declarations in K'UHUL syntax:
//   fn read_file(path:string) -> string    { effect="io" }
//   fn write_file(path:string, content:string) -> bool { effect="io" }
//   fn exec(cmd:string, args:list) -> list { effect="process" }
//   fn tool(name:string, input:map) -> map { effect="tool" }
//   fn agent(name:string, prompt:string) -> string { effect="agent" }
//   fn micronaut(name:string, args:map) -> map { effect="micronaut" }
//   fn skill(name:string, input:any) -> any { effect="skill" }
//   fn action(name:string, params:map) -> map { effect="action" }
//   fn verb(name:string, subject:string, object:string) -> string { effect="verb" }
//   fn bot(name:string, message:string) -> string { effect="bot" }
//   fn shell(cmd:string) -> bool { effect="shell" }
//   fn http(method:string, url:string, body:string) -> map { effect="network" }
//
// KXML integration:
//   <node id="call_tool" phase="Sek" domain="action">
//     <bind from="tool_name,input" to="output" transform="kuhul.tool" />
//     <effect type="tool" side_effect="true" />
//   </node>

#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace Kuhul {
namespace Functions {

// ─── KValue — dynamically-typed K'UHUL value ─────────────────────────────────

struct KValue;
using KNull   = std::nullptr_t;
using KList   = std::vector<KValue>;
using KMap    = std::map<std::string, KValue>;

struct KValue : std::variant<KNull, std::string, double, int64_t, bool, KList, KMap> {
    using Base = std::variant<KNull, std::string, double, int64_t, bool, KList, KMap>;
    using Base::Base;
    using Base::operator=;

    bool        is_null()   const { return std::holds_alternative<KNull>(*this); }
    bool        is_string() const { return std::holds_alternative<std::string>(*this); }
    bool        is_number() const { return std::holds_alternative<double>(*this) ||
                                           std::holds_alternative<int64_t>(*this); }
    bool        is_bool()   const { return std::holds_alternative<bool>(*this); }
    bool        is_list()   const { return std::holds_alternative<KList>(*this); }
    bool        is_map()    const { return std::holds_alternative<KMap>(*this); }

    std::string as_string() const { return std::get<std::string>(*this); }
    double      as_double() const {
        if (is_number()) {
            if (std::holds_alternative<double>(*this)) return std::get<double>(*this);
            return static_cast<double>(std::get<int64_t>(*this));
        }
        return 0.0;
    }
    bool        as_bool()   const { return std::get<bool>(*this); }
    const KList& as_list()  const { return std::get<KList>(*this); }
    const KMap&  as_map()   const { return std::get<KMap>(*this); }
};

using KFunction = std::function<KValue(const std::vector<KValue>&)>;

// ─── FunctionRegistry ─────────────────────────────────────────────────────────

class FunctionRegistry {
public:
    FunctionRegistry() { RegisterBuiltins(); }

    void Register(const std::string& name, KFunction fn,
                  const std::string& effect = "pure") {
        fns_[name]     = std::move(fn);
        effects_[name] = effect;
    }

    bool Has(const std::string& name) const {
        return fns_.count(name) > 0;
    }

    KValue Call(const std::string& name, const std::vector<KValue>& args) const {
        auto it = fns_.find(name);
        if (it == fns_.end())
            throw std::runtime_error("Unknown K'UHUL function: " + name);
        return it->second(args);
    }

    const std::string& Effect(const std::string& name) const {
        static const std::string none;
        auto it = effects_.find(name);
        return it != effects_.end() ? it->second : none;
    }

private:
    std::map<std::string, KFunction>    fns_;
    std::map<std::string, std::string>  effects_;

    // ── Builtins ─────────────────────────────────────────────────────────────

    void RegisterBuiltins() {

        // ── read_file(path) -> string ────────────────────────────────────────
        Register("read_file", [](const std::vector<KValue>& args) -> KValue {
            if (args.empty() || !args[0].is_string()) return KNull{};
            std::ifstream f(args[0].as_string());
            if (!f) return KNull{};
            return std::string(std::istreambuf_iterator<char>(f), {});
        }, "io");

        // ── write_file(path, content) -> bool ────────────────────────────────
        Register("write_file", [](const std::vector<KValue>& args) -> KValue {
            if (args.size() < 2 || !args[0].is_string() || !args[1].is_string())
                return false;
            std::ofstream f(args[0].as_string());
            if (!f) return false;
            f << args[1].as_string();
            return true;
        }, "io");

        // ── exec(cmd, args?) -> list<string> ─────────────────────────────────
        Register("exec", [](const std::vector<KValue>& args) -> KValue {
            if (args.empty() || !args[0].is_string()) return KList{};
            std::string cmd = args[0].as_string();
            if (args.size() > 1 && args[1].is_list())
                for (const auto& a : args[1].as_list())
                    if (a.is_string()) cmd += " " + a.as_string();

            char buf[256];
            std::string result;
#ifdef _WIN32
            std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd.c_str(), "r"), _pclose);
#else
            std::unique_ptr<FILE, decltype(&pclose)>  pipe(popen(cmd.c_str(), "r"),  pclose);
#endif
            if (!pipe) return KList{};
            while (fgets(buf, sizeof buf, pipe.get())) result += buf;

            KList lines;
            std::istringstream ss(result);
            std::string line;
            while (std::getline(ss, line)) lines.push_back(KValue{line});
            return lines;
        }, "process");

        // ── shell(cmd) -> bool ───────────────────────────────────────────────
        Register("shell", [](const std::vector<KValue>& args) -> KValue {
            if (args.empty() || !args[0].is_string()) return false;
            return std::system(args[0].as_string().c_str()) == 0;
        }, "shell");

        // ── tool(name, input) -> map ─────────────────────────────────────────
        Register("tool", [](const std::vector<KValue>& args) -> KValue {
            KMap out;
            std::string name = args.size() > 0 && args[0].is_string()
                               ? args[0].as_string() : "unknown";
            out["tool"]   = KValue{name};
            out["status"] = KValue{std::string("executed")};
            if (args.size() > 1) out["input"] = args[1];
            return out;
        }, "tool");

        // ── agent(name, prompt) -> string ────────────────────────────────────
        Register("agent", [](const std::vector<KValue>& args) -> KValue {
            std::string name   = args.size()>0 && args[0].is_string() ? args[0].as_string() : "agent";
            std::string prompt = args.size()>1 && args[1].is_string() ? args[1].as_string() : "";
            return KValue{"[" + name + "] " + prompt};
        }, "agent");

        // ── micronaut(name, args?) -> map ─────────────────────────────────────
        Register("micronaut", [](const std::vector<KValue>& args) -> KValue {
            std::string name = args.size()>0 && args[0].is_string() ? args[0].as_string() : "capsule";
            KMap out;
            out["micronaut"] = KValue{name};
            out["pid"]       = KValue{static_cast<int64_t>(std::hash<std::string>{}(name))};
            out["status"]    = KValue{std::string("spawned")};
            if (args.size() > 1) out["params"] = args[1];
            return out;
        }, "micronaut");

        // ── skill(name, input) -> any ─────────────────────────────────────────
        Register("skill", [](const std::vector<KValue>& args) -> KValue {
            KMap out;
            out["skill"] = args.size()>0 ? args[0] : KValue{KNull{}};
            out["input"] = args.size()>1 ? args[1] : KValue{KNull{}};
            out["status"] = KValue{std::string("executed")};
            return out;
        }, "skill");

        // ── action(name, params) -> map ──────────────────────────────────────
        Register("action", [](const std::vector<KValue>& args) -> KValue {
            KMap out;
            out["action"] = args.size()>0 ? args[0] : KValue{KNull{}};
            out["params"] = args.size()>1 ? args[1] : KValue{KNull{}};
            out["status"] = KValue{std::string("performed")};
            return out;
        }, "action");

        // ── verb(name, subject, object) -> string ────────────────────────────
        Register("verb", [](const std::vector<KValue>& args) -> KValue {
            std::string v = args.size()>0 && args[0].is_string() ? args[0].as_string() : "";
            std::string s = args.size()>1 && args[1].is_string() ? args[1].as_string() : "";
            std::string o = args.size()>2 && args[2].is_string() ? args[2].as_string() : "";
            return KValue{s + " " + v + " " + o};
        }, "verb");

        // ── bot(name, message) -> string ─────────────────────────────────────
        Register("bot", [](const std::vector<KValue>& args) -> KValue {
            std::string name = args.size()>0 && args[0].is_string() ? args[0].as_string() : "bot";
            std::string msg  = args.size()>1 && args[1].is_string() ? args[1].as_string() : "";
            return KValue{"[" + name + "] " + msg};
        }, "bot");

        // ── http(method, url, body) -> map ───────────────────────────────────
        Register("http", [](const std::vector<KValue>& args) -> KValue {
            KMap out;
            out["method"] = args.size()>0 ? args[0] : KValue{std::string("GET")};
            out["url"]    = args.size()>1 ? args[1] : KValue{std::string("")};
            out["status"] = KValue{static_cast<int64_t>(200)};
            out["body"]   = KValue{std::string("(http stub)")};
            return out;
        }, "network");
    }
};

// ─── Global singleton ─────────────────────────────────────────────────────────

inline FunctionRegistry& GlobalRegistry() {
    static FunctionRegistry inst;
    return inst;
}

inline KValue Call(const std::string& name, const std::vector<KValue>& args) {
    return GlobalRegistry().Call(name, args);
}

} // namespace Functions
} // namespace Kuhul
