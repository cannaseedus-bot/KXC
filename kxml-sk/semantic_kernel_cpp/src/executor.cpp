#include "../include/executor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>


// ---------------- Execution helpers ----------------
XCFEPhase parsePhase(const std::string &s){
    if (s=="@Pop"||s=="Pop") return XCFEPhase::Pop;
    if (s=="@Wo"||s=="Wo") return XCFEPhase::Wo;
    if (s=="@Sek"||s=="Sek") return XCFEPhase::Sek;
    if (s=="@Ch'en"||s=="Chen"||s=="Ch'en") return XCFEPhase::Chen;
    if (s=="@Xul"||s=="Xul") return XCFEPhase::Xul;
    return XCFEPhase::Unknown;
}

void ExecutionContext::log(const std::string &m){
    logs.push_back(m);
    std::cout<<m<<"\n";
}

void OpRegistry::registerOp(const std::string &name, OpHandler handler){ ops[name]=handler; }
bool OpRegistry::execute(const std::string &name, ExecutionNode &node, ExecutionContext &ctx) const{
    auto it = ops.find(name);
    if (it==ops.end()){ ctx.log("Unknown op: "+name); return false; }
    it->second(node, ctx);
    return true;
}

// ---------------- Simple XML parser (local copy) ----------------
struct XmlNode { std::string name; std::string text; std::vector<XmlNode> children; };
static void parse_simple_xml(const std::string &s, XmlNode &root){
    size_t i=0,n=s.size(); std::vector<XmlNode*> stack; root.name="_root_"; stack.push_back(&root);
    while(i<n){
        if (s[i]=='<'){
            if (i+1<n && s[i+1]=='/'){ size_t j=s.find('>',i+2); if (j==std::string::npos) break; if (stack.size()>1) stack.pop_back(); i=j+1; }
            else { size_t j=s.find('>',i+1); if (j==std::string::npos) break; std::string tag=s.substr(i+1,j-(i+1)); size_t sp=tag.find(' '); if (sp!=std::string::npos) tag=tag.substr(0,sp); XmlNode node; node.name=tag; stack.back()->children.push_back(node); stack.push_back(&stack.back()->children.back()); i=j+1; size_t k=s.find('<',i); if (k!=std::string::npos && k>i){ std::string txt=s.substr(i,k-i); auto l=txt.find_first_not_of("\n\t \r"); auto r=txt.find_last_not_of("\n\t \r"); if (l!=std::string::npos && r!=std::string::npos) stack.back()->text=txt.substr(l,r-l+1); i=k; } }
        } else i++; }
}

static void xml_collect_nodes_by_name(const std::vector<XmlNode*> &nodes, const std::string &name, std::vector<XmlNode*> &out){
    for (auto n: nodes) for (auto &c: n->children) if (c.name==name) out.push_back(&const_cast<XmlNode&>(c));
}

// parse <node> XML into ExecutionNode structure
static bool xmlnode_to_executionnode(const XmlNode &xn, ExecutionNode &out){
    // expect children: phase, op, state (entries), children (node...)
    for (auto &c: xn.children){
        if (c.name=="phase") out.phase = parsePhase(c.text);
        else if (c.name=="op") out.op = c.text;
        else if (c.name=="state"){
            for (auto &e: c.children){ // entry
                std::string key; std::string val;
                for (auto &kv: e.children){ if (kv.name=="key") key=kv.text; if (kv.name=="value") val=kv.text; }
                // simplest: parse integer else string
                if (!key.empty()){
                    bool isnum=true; for (char ch: val) if (!isdigit((unsigned char)ch) && ch!='-') { isnum=false; break; }
                    if (isnum && !val.empty()) out.state[key] = int64_t(std::stoll(val)); else out.state[key] = val;
                }
            }
        } else if (c.name=="children"){
            for (auto &childnode: c.children){ if (childnode.name=="node"){ ExecutionNode ch; xmlnode_to_executionnode(childnode, ch); out.children.push_back(std::move(ch)); } }
        }
    }
    return true;
}

bool parse_execution_xml(const std::string &path, ExecutionNode &out_root){
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::stringstream ss; ss<<ifs.rdbuf();
    std::string s = ss.str();
    XmlNode root; parse_simple_xml(s, root);
    // find first node element under root
    for (auto &c: root.children){ if (c.name=="node"){ return xmlnode_to_executionnode(c, out_root); } }
    return false;
}

// collect nodes in deterministic preorder
void collectPhaseNodes(ExecutionNode &node, XCFEPhase phase, std::vector<ExecutionNode*> &out){
    if (node.phase==phase) out.push_back(&node);
    for (auto &ch: node.children) collectPhaseNodes(ch, phase, out);
}

// register default ops
void registerCoreOps(OpRegistry &reg){
    reg.registerOp("init", [](ExecutionNode&, ExecutionContext& ctx){ ctx.log("[Pop] init"); });
    reg.registerOp("assign", [](ExecutionNode &node, ExecutionContext &ctx){
        for (auto &p: node.state){ ctx.phaseBuffer[p.first] = p.second; ctx.log("[Wo] set "+p.first); }
    });
    reg.registerOp("add", [](ExecutionNode &node, ExecutionContext &ctx){
        int64_t a=0,b=0; if (node.state.count("a")) a=std::get<int64_t>(node.state.at("a")); if (node.state.count("b")) b=std::get<int64_t>(node.state.at("b")); ctx.phaseBuffer["result"] = a+b; ctx.log("[Sek] add executed");
    });
    reg.registerOp("emit", [](ExecutionNode&, ExecutionContext& ctx){
        if (ctx.global.count("result")){
            auto &v = ctx.global.at("result");
            if (std::holds_alternative<int64_t>(v)) ctx.log("[Chen] result="+std::to_string(std::get<int64_t>(v)));
            else ctx.log("[Chen] result (non-int)");
        } else ctx.log("[Chen] no result");
    });
    reg.registerOp("cleanup", [](ExecutionNode&, ExecutionContext& ctx){ ctx.log("[Xul] cleanup"); });
}

bool run_execution(ExecutionNode &root, OpRegistry &reg, size_t threads){
    ThreadPool pool(threads);
    ExecutionContext ctx;
    // phases ordered
    std::vector<XCFEPhase> phases = {XCFEPhase::Pop, XCFEPhase::Wo, XCFEPhase::Sek, XCFEPhase::Chen, XCFEPhase::Xul};
    for (auto phase: phases){
        // collect
        std::vector<ExecutionNode*> nodes; collectPhaseNodes(root, phase, nodes);
        // deterministic ordering preserved from traversal
        // dispatch tasks
        for (auto n: nodes){ pool.taskStarted(); pool.enqueue([&reg,n,&ctx,&pool](){ if (!n->op.empty()) reg.execute(n->op, *n, ctx); pool.taskFinished(); }); }
        // barrier
        pool.wait();
        // merge phaseBuffer into global (phase-local state model)
        for (auto &p: ctx.phaseBuffer) ctx.global[p.first] = p.second;
        ctx.phaseBuffer.clear();
    }
    return true;
}
