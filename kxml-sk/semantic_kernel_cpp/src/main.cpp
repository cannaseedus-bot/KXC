#include <iostream>
#include <cstdlib>
#include "../include/semantic_kernel.h"
#include "../include/compiler.h"
#include "../include/planner.h"
#include "../include/memory.h"
#include "../include/dx12_executor.h"
#include "../include/runtime_loop.h"
#include "../include/semantic_reader.h"

extern "C" {
    void ingestFact_c(const char* id, const char* text, const float* embedding, size_t emb_len);
    int queryFacts_c(const float* embedding, size_t emb_len, int k);
}

int main(int argc, char** argv){

    if (argc<2){
        std::cout<<"Usage: semantic_kernel_cli <command> [args]\n";
        std::cout<<"Commands: detect <text>, parse <text> <lang?>, compile_ir <text> <out.json>, compile_ir_xml <text> <out.xml>, read_topology <file> <out.json>, absorb_jsonl <file.jsonl> <out.json>, query_xml <file> <xpath>\n";
        return 1;
    }
    std::string cmd = argv[1];
    SemanticKernel sk;
    if (cmd=="detect"){
        std::string text;
        for (int i=2;i<argc;++i){ if (i>2) text += " "; text += argv[i]; }
        auto r = sk.detect_language(text);
        std::cout<<"language="<<r.first<<" confidence="<<r.second<<"\n";
        return 0;
    }
    if (cmd=="parse"){
        std::string text;
        for (int i=2;i<argc;++i){ if (i>2) text += " "; text += argv[i]; }
        auto clause = sk.parse_to_semantics(text);
        std::cout<<"predicate="<<clause.predicate<<"\n";
        for (auto &a: clause.arguments) std::cout<<a.role<<":"<<a.entity<<"\n";
        return 0;
    }
    if (cmd=="compile_ir"){
        if (argc<4){ std::cerr<<"compile_ir requires <text> <out.json>\n"; return 2; }
        std::string text = argv[2];
        std::string out = argv[3];
        auto clause = sk.parse_to_semantics(text);
        if (sk.write_ir_json(clause, out)){
            std::cout<<"wrote "<<out<<"\n";
            return 0;
        } else {
            std::cerr<<"failed to write "<<out<<"\n";
            return 3;
        }
    }
    if (cmd=="compile_ir_xml"){
        if (argc<4){ std::cerr<<"compile_ir_xml requires <text> <out.xml>\n"; return 2; }
        std::string text = argv[2];
        std::string out = argv[3];
        auto clause = sk.parse_to_semantics(text);
        if (sk.write_ir_xml(clause, out)){
            std::cout<<"wrote "<<out<<"\n";
            return 0;
        } else {
            std::cerr<<"failed to write "<<out<<"\n";
            return 3;
        }
    }
    if (cmd=="compile_ir_protobuf"){
        if (argc<4){ std::cerr<<"compile_ir_protobuf requires <text> <out.pb>\n"; return 2; }
        std::string text = argv[2];
        std::string out = argv[3];
        auto clause = sk.parse_to_semantics(text);
        if (sk.write_ir_protobuf(clause, out)){
            std::cout<<"wrote "<<out<<"\n";
            return 0;
        } else {
            std::cerr<<"failed to write "<<out<<" (Protobuf support may be missing)\n";
            return 3;
        }
    }
    if (cmd=="execute_ir"){
        if (argc<3){ std::cerr<<"execute_ir requires <ir.xml> (XML execution IR)\n"; return 2; }
        std::string file = argv[2];
        // currently only XML execution format supported
        ExecutionNode root;
        if (!parse_execution_xml(file, root)){
            std::cerr<<"failed to parse execution IR (xml required)\n"; return 4;
        }
        OpRegistry reg; registerCoreOps(reg);
        if (!run_execution(root, reg, std::thread::hardware_concurrency())){
            std::cerr<<"execution failed\n"; return 5;
        }
        return 0;
    }
    if (cmd=="execute_ir_batched"){
        if (argc<3){ std::cerr<<"execute_ir_batched requires <ir.xml>\n"; return 2; }
        std::string file = argv[2];
        ExecutionNode root;
        if (!parse_execution_xml(file, root)) { std::cerr<<"failed to parse execution IR (xml required)\n"; return 4; }
        OpRegistry reg; registerCoreOps(reg);
        // batched SIMD execution path
        execute_phases_with_batches(root, reg, std::thread::hardware_concurrency());
        // Note: batched executor currently updates node.state["result"] but does not run non-SIMD ops.
        return 0;
    }
    if (cmd=="compile_plan_execute"){
        if (argc<3){ std::cerr<<"compile_plan_execute requires <ir.xml>\n"; return 2; }
        std::string file = argv[2];
        ExecutionNode root;
        if (!parse_execution_xml(file, root)) { std::cerr<<"failed to parse execution IR (xml required)\n"; return 4; }
        CompiledProgram prog;
        if (!compileProgram(root, prog)) { std::cerr<<"compile failed\n"; return 5; }
        auto plan = planProgram(prog);
        executePlan(plan, prog.buffers, std::thread::hardware_concurrency());
        // after execution, materialize outputs into root state (best-effort)
        std::cout<<"plan executed\n";
        return 0;
    }
    if (cmd=="execute_plan_dx12"){
        if (argc<3){ std::cerr<<"execute_plan_dx12 requires <ir.xml> [schedule_out.json]\n"; return 2; }
        std::string file = argv[2];
        std::string out = (argc>=4)? argv[3] : std::string("gpu_dispatch_schedule.json");
        ExecutionNode root;
        if (!parse_execution_xml(file, root)) { std::cerr<<"failed to parse execution IR (xml required)\n"; return 4; }
        CompiledProgram prog;
        if (!compileProgram(root, prog)) { std::cerr<<"compile failed\n"; return 5; }
        auto plan = planProgram(prog);
        if (!executePlanDX12(plan, prog, out, /*enableGpu=*/false)) { std::cerr<<"DX12 execution failed\n"; return 6; }
        std::cout<<"DX12 schedule generated: "<<out<<"\n";
        return 0;
    }
    if (cmd=="stream_decode"){
        // simple streaming run for debug
        return main_debug_stream(argc, argv);
    }

    if (cmd=="read_topology"){
        if (argc<4){ std::cerr<<"read_topology requires <input.xml|toml|txt> <out.json> [threshold]\n"; return 2; }
        std::string file = argv[2];
        std::string out = argv[3];
        float threshold = (argc>=5) ? static_cast<float>(std::atof(argv[4])) : 0.35f;
        if (!write_semantic_reader_report(file, out, threshold)){
            std::cerr<<"failed to read topology or write "<<out<<"\n";
            return 4;
        }
        std::cout<<"semantic reader report generated: "<<out<<"\n";
        return 0;
    }

    if (cmd=="absorb_jsonl"){
        if (argc<4){ std::cerr<<"absorb_jsonl requires <input.jsonl> <out.json> [threshold]\n"; return 2; }
        std::string file = argv[2];
        std::string out = argv[3];
        float threshold = (argc>=5) ? static_cast<float>(std::atof(argv[4])) : 0.35f;
        if (!write_semantic_jsonl_absorb_report(file, out, threshold)){
            std::cerr<<"failed to absorb jsonl or write "<<out<<"\n";
            return 4;
        }
        std::cout<<"semantic jsonl absorb report generated: "<<out<<"\n";
        return 0;
    }

    // Memory integration helpers (simple CLI tests)
    if (cmd=="mem_ingest"){
        // Usage: mem_ingest [id] [text]
        std::string id = (argc>=3)? argv[2] : std::string("m1");
        std::string text = (argc>=4)? argv[3] : std::string("test memory");
        // create a dummy embedding (8 dims)
        float emb[8] = {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f};
        ingestFact_c(id.c_str(), text.c_str(), emb, 8);
        std::cout<<"ingested memory id="<<id<<"\n";
        return 0;
    }

    if (cmd=="mem_query"){
        // Usage: mem_query [k]
        int k = (argc>=3) ? std::atoi(argv[2]) : 5;
        float emb[8] = {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f};
        int found = queryFacts_c(emb, 8, k);
        std::cout<<"memory query returned "<<found<<" candidates\n";
        return 0;
    }
    if (cmd=="query_xml"){
        if (argc<4){ std::cerr<<"query_xml requires <file> <xpath>\n"; return 2; }
        std::string file = argv[2];
        std::string xpath = argv[3];
        std::vector<std::string> results;
        if (!sk.query_ir_xml(file, xpath, results)){
            std::cerr<<"query failed\n"; return 4;
        }
        for (auto &r: results) std::cout<<r<<"\n";
        return 0;
    }
    std::cerr<<"unknown command"<<std::endl;
    return 10;
}
