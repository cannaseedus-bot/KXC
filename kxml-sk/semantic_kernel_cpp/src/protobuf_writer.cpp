#include "../include/semantic_kernel.h"
#include <fstream>

#ifdef USE_PROTOBUF
#include "ir.pb.h"

bool SemanticKernel::write_ir_protobuf(const SemanticClause &clause, const std::string &path){
    semantic::Clause msg;
    msg.set_predicate(clause.predicate);
    msg.set_tense(clause.tense);
    msg.set_polarity(clause.polarity);
    for (const auto &a: clause.arguments){
        auto *arg = msg.add_arguments();
        arg->set_role(a.role);
        arg->set_entity(a.entity);
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    return msg.SerializeToOstream(&ofs);
}

#else
bool SemanticKernel::write_ir_protobuf(const SemanticClause&, const std::string&){
    return false;
}
#endif
