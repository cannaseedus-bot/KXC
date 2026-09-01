#include "../include/semantic_kernel.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

#include "tokenizer.h"

TinyTokenizer g_tokenizer;

SemanticKernel::SemanticKernel() {}

static std::string to_lower(std::string s){
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

std::pair<std::string,double> SemanticKernel::detect_language(const std::string &text){
    // naive heuristics: presence of specific characters/words
    std::string low = to_lower(text);
    if (low.find("的")!=std::string::npos || low.find("了")!=std::string::npos) return {"mandarin", 0.9};
    if (low.find("は")!=std::string::npos || low.find("です")!=std::string::npos) return {"japanese", 0.9};
    if (low.find("the ")!=std::string::npos || low.find(" and ")!=std::string::npos) return {"english", 0.85};
    return {"unknown", 0.5};
}

SemanticClause SemanticKernel::parse_to_semantics(const std::string &utterance, const std::string &lang){
    // Very small heuristic parser: split by spaces, find first noun-like/verb-like tokens
    SemanticClause out;
    out.tense = "present";
    out.polarity = "affirmative";
    auto toks = g_tokenizer.tokenize(utterance);

    // verb detection via lemmatization and simple heuristics
    std::string verb="";
    for (size_t i=0;i<toks.size();++i){
        std::string lem = g_tokenizer.lemmatize(toks[i]);
        if (lem=="is"||lem=="are"||lem=="run"||lem=="move"||lem=="eat"||lem=="translate") { verb = lem; break; }
        // prefer first verb-like candidate (naive)
        if (i>0 && toks[i].back()=='s') { verb = lem; break; }
    }
    if (verb.empty() && toks.size()>1) verb = g_tokenizer.lemmatize(toks[1]);
    out.predicate = verb.empty()?"be":verb;

    // subject = first token
    if (!toks.empty()){
        SemanticArgument a;
        a.role = "agent";
        a.entity = toks[0];
        out.arguments.push_back(a);
    }
    // object = third token if exists
    if (toks.size()>2){
        SemanticArgument b; b.role = "patient"; b.entity = toks[2];
        out.arguments.push_back(b);
    }
    return out;
}

bool SemanticKernel::write_ir_json(const SemanticClause &clause, const std::string &path){
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << "{\n";
    ofs << "  \"predicate\": \"" << clause.predicate << "\",\n";
    ofs << "  \"tense\": \"" << clause.tense << "\",\n";
    ofs << "  \"polarity\": \"" << clause.polarity << "\",\n";
    ofs << "  \"arguments\": [\n";
    for (size_t i=0;i<clause.arguments.size();++i){
        ofs << "    { \"role\": \""<< clause.arguments[i].role << "\", \"entity\": \""<< clause.arguments[i].entity << "\" }";
        if (i+1<clause.arguments.size()) ofs<<",";
        ofs<<"\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    return true;
}

static std::string xml_escape(const std::string &s){
    std::string out;
    for (char c: s){
        switch(c){
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += c; break;
        }
    }
    return out;
}

#ifdef USE_PUGIXML
#include <pugixml.hpp>

bool SemanticKernel::write_ir_xml(const SemanticClause &clause, const std::string &path){
    pugi::xml_document doc;
    auto root = doc.append_child("clause");
    root.append_child("predicate").text().set(clause.predicate.c_str());
    root.append_child("tense").text().set(clause.tense.c_str());
    root.append_child("polarity").text().set(clause.polarity.c_str());
    auto args = root.append_child("arguments");
    for (const auto &arg: clause.arguments){
        auto a = args.append_child("argument");
        a.append_child("role").text().set(arg.role.c_str());
        a.append_child("entity").text().set(arg.entity.c_str());
    }
    return doc.save_file(path.c_str());
}

bool SemanticKernel::query_ir_xml(const std::string &xml_path, const std::string &query, std::vector<std::string> &results){
    pugi::xml_document doc;
    pugi::xml_parse_result res = doc.load_file(xml_path.c_str());
    if (!res) return false;
    try {
        pugi::xpath_node_set nodes = doc.select_nodes(query.c_str());
        for (auto &n: nodes){
            results.push_back(n.node().text().as_string());
        }
        return true;
    } catch (...) {
        return false;
    }
}

#else

// fallback simple XML writer + query implementation (no external deps)

bool SemanticKernel::write_ir_xml(const SemanticClause &clause, const std::string &path){
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << "<clause>\n";
    ofs << "  <predicate>"<< xml_escape(clause.predicate) << "</predicate>\n";
    ofs << "  <tense>"<< xml_escape(clause.tense) << "</tense>\n";
    ofs << "  <polarity>"<< xml_escape(clause.polarity) << "</polarity>\n";
    ofs << "  <arguments>\n";
    for (const auto &arg: clause.arguments){
        ofs << "    <argument>\n";
        ofs << "      <role>"<< xml_escape(arg.role) << "</role>\n";
        ofs << "      <entity>"<< xml_escape(arg.entity) << "</entity>\n";
        ofs << "    </argument>\n";
    }
    ofs << "  </arguments>\n";
    ofs << "</clause>\n";
    return true;
}

// Very small XML DOM for simple querying
struct XmlNode {
    std::string name;
    std::string text;
    std::vector<XmlNode> children;
};

static void parse_simple_xml(const std::string &s, XmlNode &root){
    size_t i=0, n=s.size();
    std::vector<XmlNode*> stack;
    root.name = "_root_";
    stack.push_back(&root);
    while (i<n){
        if (s[i]=='<'){
            if (i+1<n && s[i+1]=='/'){
                // closing tag
                size_t j = s.find('>', i+2);
                if (j==std::string::npos) break;
                // pop
                if (stack.size()>1) stack.pop_back();
                i = j+1;
            } else {
                // opening tag
                size_t j = s.find('>', i+1);
                if (j==std::string::npos) break;
                std::string tag = s.substr(i+1, j-(i+1));
                // remove attributes (not used)
                size_t sp = tag.find(' ');
                if (sp!=std::string::npos) tag = tag.substr(0, sp);
                XmlNode node;
                node.name = tag;
                stack.back()->children.push_back(node);
                stack.push_back(&stack.back()->children.back());
                i = j+1;
                // gather text until next '<'
                size_t k = s.find('<', i);
                if (k!=std::string::npos && k>i){
                    std::string txt = s.substr(i, k-i);
                    // trim whitespace
                    auto l = txt.find_first_not_of("\n\t \r");
                    auto r = txt.find_last_not_of("\n\t \r");
                    if (l!=std::string::npos && r!=std::string::npos){
                        stack.back()->text = txt.substr(l, r-l+1);
                    }
                    i = k;
                }
            }
        } else {
            i++;
        }
    }
}

static void collect_by_path(const std::vector<XmlNode*> &nodes, const std::string &token, const std::string &pred_name, const std::string &pred_val, std::vector<XmlNode*> &out){
    for (auto n: nodes){
        for (auto &c: n->children){
            if (c.name == token){
                if (pred_name.empty()) out.push_back(&const_cast<XmlNode&>(c));
                else {
                    // check predicate: find child pred_name with text == pred_val
                    bool ok=false;
                    for (auto &cc: c.children){ if (cc.name==pred_name && cc.text==pred_val) { ok=true; break; } }
                    if (ok) out.push_back(&const_cast<XmlNode&>(c));
                }
            }
        }
    }
}

bool SemanticKernel::query_ir_xml(const std::string &xml_path, const std::string &query, std::vector<std::string> &results){
    std::ifstream ifs(xml_path);
    if (!ifs) return false;
    std::stringstream ss; ss << ifs.rdbuf();
    std::string s = ss.str();
    XmlNode root;
    parse_simple_xml(s, root);
    // parse query like /a/b/c or with predicate element[child='val']
    std::vector<std::string> parts;
    size_t p=0;
    while (p<query.size()){
        if (query[p]=='/') { p++; continue; }
        size_t q = query.find('/', p);
        if (q==std::string::npos) q = query.size();
        parts.push_back(query.substr(p, q-p));
        p = q;
    }
    std::vector<XmlNode*> current = { &root };
    for (auto &part: parts){
        // check predicate
        std::string token = part;
        std::string pred_name="", pred_val="";
        size_t br = part.find('[');
        if (br!=std::string::npos){
            token = part.substr(0, br);
            size_t brc = part.find(']', br+1);
            if (brc!=std::string::npos){
                std::string inside = part.substr(br+1, brc-br-1); // e.g., role='agent' or child='val'
                size_t eq = inside.find('=');
                if (eq!=std::string::npos){
                    pred_name = inside.substr(0, eq);
                    pred_val = inside.substr(eq+1);
                    // strip quotes
                    if (!pred_val.empty() && (pred_val.front()=='\'' || pred_val.front()=='\"')) pred_val = pred_val.substr(1, pred_val.size()-2);
                }
            }
        }
        std::vector<XmlNode*> next;
        collect_by_path(current, token, pred_name, pred_val, next);
        current = next;
    }
    // collect text of current nodes or their children
    for (auto n: current){
        if (!n->text.empty()) results.push_back(n->text);
        else {
            // collect entity/text children
            for (auto &c: n->children){ if (!c.text.empty()) results.push_back(c.text); }
        }
    }
    return true;
}
#endif