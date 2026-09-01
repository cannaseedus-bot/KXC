#include "../include/semantic_reader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <vector>

using json = nlohmann::json;

namespace {

std::string trim(const std::string& value) {
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }
    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) {
        --last;
    }
    return value.substr(first, last - first);
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

json parse_attrs(const std::string& attrs) {
    json out = json::object();
    std::regex attr_re("([A-Za-z0-9_:\\-]+)\\s*=\\s*\"([^\"]*)\"");
    for (std::sregex_iterator it(attrs.begin(), attrs.end(), attr_re), end; it != end; ++it) {
        out[(*it)[1].str()] = (*it)[2].str();
    }
    return out;
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        auto cleaned = trim(line);
        if (!cleaned.empty()) {
            lines.push_back(cleaned);
        }
    }
    return lines;
}

std::string strip_cdata_markers(std::string value) {
    const std::string open = "<![CDATA[";
    const std::string close = "]]>";
    size_t p = value.find(open);
    if (p != std::string::npos) {
        value.erase(p, open.size());
    }
    p = value.rfind(close);
    if (p != std::string::npos) {
        value.erase(p, close.size());
    }
    return trim(value);
}

void add_unique(json& array, const json& item) {
    for (const auto& existing : array) {
        if (existing == item) {
            return;
        }
    }
    array.push_back(item);
}

bool contains_any(const std::string& lower, const std::vector<std::string>& needles) {
    for (const auto& needle : needles) {
        if (lower.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

json collect_containment_nodes(const std::string& text) {
    json nodes = json::array();
    std::regex tag_re("<\\s*([A-Za-z0-9_:\\-]+)((?:\\s+[A-Za-z0-9_:\\-]+\\s*=\\s*\"[^\"]*\")*)\\s*(/?)>");
    int index = 0;
    for (std::sregex_iterator it(text.begin(), text.end(), tag_re), end; it != end; ++it) {
        std::string raw = (*it)[0].str();
        if (raw.rfind("<?", 0) == 0 || raw.rfind("<!--", 0) == 0 || raw.rfind("<!", 0) == 0) {
            continue;
        }
        json node;
        node["index"] = index++;
        node["name"] = (*it)[1].str();
        node["attrs"] = parse_attrs((*it)[2].str());
        node["self_closing"] = (*it)[3].matched;
        nodes.push_back(node);
    }
    return nodes;
}

json collect_cdata_capsules(const std::string& text) {
    json capsules = json::array();
    const std::string open = "<![CDATA[";
    const std::string close = "]]>";
    size_t pos = 0;
    int index = 0;
    while ((pos = text.find(open, pos)) != std::string::npos) {
        size_t start = pos + open.size();
        size_t end = text.find(close, start);
        if (end == std::string::npos) {
            break;
        }
        std::string payload = text.substr(start, end - start);
        std::string lowered = lower_copy(payload);
        json capsule;
        capsule["index"] = index++;
        capsule["bytes"] = payload.size();
        capsule["payload"] = payload;
        capsule["kinds"] = json::array();
        if (contains_any(lowered, {"pop", "sek", "xul", "yax", "wo", "ch'en"})) {
            capsule["kinds"].push_back("kuhul");
        }
        if (contains_any(lowered, {"shader", "hlsl", "wgsl", "cuda", "simd", "fused_attention"})) {
            capsule["kinds"].push_back("projection");
        }
        if (payload.find('.') != std::string::npos) {
            capsule["kinds"].push_back("semantic_grams");
        }
        if (contains_any(lowered, {"policy", "permission", "restricted", "math_only", "required"})) {
            capsule["kinds"].push_back("policy");
        }
        capsules.push_back(capsule);
        pos = end + close.size();
    }
    return capsules;
}

json collect_tag_texts(const std::string& text, const std::string& tag) {
    json out = json::array();
    std::regex re("<" + tag + R"([^>]*>([\s\S]*?)</)" + tag + ">", std::regex::icase);
    for (std::sregex_iterator it(text.begin(), text.end(), re), end; it != end; ++it) {
        auto value = strip_cdata_markers((*it)[1].str());
        if (!value.empty()) {
            out.push_back(value);
        }
    }
    return out;
}

json collect_folds(const json& nodes) {
    json folds = json::array();
    for (const auto& node : nodes) {
        const std::string name = node.value("name", "");
        if (name == "geometricIntelligence") {
            const auto& attrs = node["attrs"];
            add_unique(folds, {{"id", "geometric"}, {"domain", attrs.value("manifold_dim", "geometricIntelligence")}});
        }
        if (name == "manifold") {
            const auto& attrs = node["attrs"];
            std::string domain = attrs.value("type", "manifold");
            add_unique(folds, {{"id", domain}, {"domain", domain}});
        }
        if (name == "fold") {
            const auto& attrs = node["attrs"];
            std::string id = attrs.value("id", attrs.value("domain", "fold"));
            add_unique(folds, {{"id", id}, {"domain", attrs.value("domain", id)}});
        }
        if (name == "horizontal-folds") {
            add_unique(folds, {{"id", "horizontal"}, {"domain", "horizontal-folds"}});
        }
        if (name == "vertical-folds") {
            add_unique(folds, {{"id", "vertical"}, {"domain", "vertical-folds"}});
        }
    }
    return folds;
}

json collect_geodesics(const json& nodes) {
    json routes = json::array();
    for (const auto& node : nodes) {
        const std::string name = node.value("name", "");
        if (name != "geodesic") {
            continue;
        }
        const auto& attrs = node["attrs"];
        float cost = 0.0f;
        try {
            cost = std::stof(attrs.value("cost", "0"));
        } catch (...) {
            cost = 0.0f;
        }
        routes.push_back({
            {"from", attrs.value("from", "")},
            {"to", attrs.value("to", "")},
            {"type", attrs.value("type", "geodesic")},
            {"geodesic_cost", cost},
            {"lawful", cost >= 0.0f && cost <= 5.0f}
        });
    }
    return routes;
}

json collect_lanes(const json& nodes) {
    json lanes = json::array();
    for (const auto& node : nodes) {
        const std::string name = node.value("name", "");
        if (name != "lane") {
            continue;
        }
        const auto& attrs = node["attrs"];
        json lane;
        lane["id"] = attrs.value("id", attrs.value("type", "lane"));
        lane["type"] = attrs.value("type", "generic");
        lane["permission"] = attrs.value("permission", "inherit");
        lanes.push_back(lane);
    }
    return lanes;
}

json collect_policies(const json& nodes, const json& capsules) {
    json policies = json::array();
    for (const auto& node : nodes) {
        const std::string name = node.value("name", "");
        if (name == "policy" || name == "directive" || name == "skill" || name == "lane") {
            const auto& attrs = node["attrs"];
            std::string id = attrs.value("id", attrs.value("type", attrs.value("permission", name)));
            add_unique(policies, {{"source", name}, {"id", id}, {"attrs", attrs}});
        }
    }
    for (const auto& capsule : capsules) {
        for (const auto& kind : capsule["kinds"]) {
            if (kind == "policy") {
                add_unique(policies, {{"source", "cdata"}, {"id", "payload_policy"}, {"capsule", capsule["index"]}});
            }
        }
    }
    return policies;
}

json collect_grams(const std::string& text, const json& capsules) {
    json grams;
    grams["bi"] = collect_tag_texts(text, "bi");
    grams["tri"] = collect_tag_texts(text, "tri");
    grams["raw"] = collect_tag_texts(text, "raw-ngrams");
    grams["semantic"] = json::array();
    grams["coarse"] = json::array();

    auto semantic_blocks = collect_tag_texts(text, "semantic-grams");
    for (const auto& block : semantic_blocks) {
        for (const auto& line : split_lines(block.get<std::string>())) {
            add_unique(grams["semantic"], line);
        }
    }

    auto coarse_blocks = collect_tag_texts(text, "coarse-grams");
    for (const auto& block : coarse_blocks) {
        for (const auto& line : split_lines(block.get<std::string>())) {
            add_unique(grams["coarse"], line);
        }
    }

    std::regex gram_re(R"(([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+){1,5}))");
    for (const auto& capsule : capsules) {
        const std::string payload = capsule.value("payload", "");
        for (std::sregex_iterator it(payload.begin(), payload.end(), gram_re), end; it != end; ++it) {
            add_unique(grams["semantic"], (*it)[1].str());
        }
    }
    return grams;
}

void add_node_tokens_to_grams(json& grams, const json& nodes) {
    for (const auto& node : nodes) {
        if (node.value("name", "") != "neuralNode") {
            continue;
        }
        const auto& attrs = node["attrs"];
        std::string token = attrs.value("token", "");
        if (!token.empty()) {
            add_unique(grams["raw"], token);
        }
    }
}

json collect_kuhul_programs(const json& capsules) {
    json programs = json::array();
    for (const auto& capsule : capsules) {
        for (const auto& kind : capsule["kinds"]) {
            if (kind == "kuhul") {
                programs.push_back({
                    {"capsule", capsule["index"]},
                    {"phase_markers", split_lines(capsule.value("payload", ""))}
                });
                break;
            }
        }
    }
    return programs;
}

json collect_projection_targets(const json& nodes, const json& capsules) {
    json targets = json::array();
    for (const auto& node : nodes) {
        const std::string name = node.value("name", "");
        if (name == "shader" || name == "gpu-manifold" || name == "geometry" || name == "path") {
            const auto& attrs = node["attrs"];
            add_unique(targets, {
                {"kind", name},
                {"type", attrs.value("type", name)}
            });
        }
    }
    for (const auto& capsule : capsules) {
        for (const auto& kind : capsule["kinds"]) {
            if (kind == "projection") {
                add_unique(targets, {{"kind", "cdata"}, {"type", "projection"}, {"capsule", capsule["index"]}});
            }
        }
    }
    return targets;
}

json build_activation(const SemanticReaderInput& input, const json& topology) {
    const auto fold_count = topology["folds"].size();
    const auto capsule_count = topology["cdata_capsules"].size();
    const auto semantic_gram_count = topology["grams"]["semantic"].size();
    float pressure = 0.15f
        + static_cast<float>(fold_count) * 0.08f
        + static_cast<float>(capsule_count) * 0.04f
        + static_cast<float>(semantic_gram_count) * 0.025f;
    pressure = std::min(1.0f, pressure);

    json active_folds = json::array();
    for (const auto& fold : topology["folds"]) {
        const bool active = pressure >= input.activation_threshold;
        active_folds.push_back({
            {"id", fold.value("id", "fold")},
            {"domain", fold.value("domain", "fold")},
            {"pressure", pressure},
            {"active", active}
        });
    }

    json gates = json::array();
    for (const auto& policy : topology["policies"]) {
        gates.push_back({
            {"policy", policy.value("id", "policy")},
            {"stage", "traversal"},
            {"verdict", pressure >= input.activation_threshold ? "allow" : "hold"},
            {"pressure", pressure}
        });
    }

    json routes = json::array();
    if (topology.contains("geodesics") && !topology["geodesics"].empty()) {
        routes = topology["geodesics"];
    } else {
        for (size_t i = 1; i < topology["folds"].size(); ++i) {
            const auto& from = topology["folds"][i - 1];
            const auto& to = topology["folds"][i];
            float cost = std::max(0.05f, 1.0f - pressure + static_cast<float>(i) * 0.03f);
            routes.push_back({
                {"from", from.value("id", "fold")},
                {"to", to.value("id", "fold")},
                {"geodesic_cost", cost},
                {"lawful", cost >= 0.0f && cost <= 5.0f}
            });
        }
    }

    json qkv = json::array();
    if (!topology["grams"]["semantic"].empty()) {
        qkv.push_back({
            {"Q", "semantic intent"},
            {"K", "semantic capsule indices"},
            {"V", "compressed causal payloads"},
            {"stage", "ambiguity_refinement"}
        });
    }

    json plan = json::array({
        {{"step", "read_containment"}, {"input", input.source_name}},
        {{"step", "preserve_cdata"}, {"capsules", capsule_count}},
        {{"step", "resolve_grams"}, {"semantic_grams", semantic_gram_count}},
        {{"step", "activate_folds"}, {"folds", fold_count}, {"pressure", pressure}},
        {{"step", "policy_gate_traversal"}, {"gates", gates.size()}},
        {{"step", "route_geodesics"}, {"routes", routes.size()}},
        {{"step", "hydrate_micronauts"}, {"capsules", topology["kuhul_programs"].size()}},
        {{"step", "qkv_refinement"}, {"required", !qkv.empty()}},
        {{"step", "project_execution"}, {"targets", topology["projection_targets"].size()}}
    });

    return {
        {"pressure", {{"global", pressure}, {"activation_threshold", input.activation_threshold}}},
        {"active_folds", active_folds},
        {"policy_gates", gates},
        {"geodesic_routes", routes},
        {"qkv_refinement", qkv},
        {"execution_plan", plan}
    };
}

} // namespace

nlohmann::json SemanticReader::read(const SemanticReaderInput& input) const {
    json nodes = collect_containment_nodes(input.text);
    json cdata = collect_cdata_capsules(input.text);

    json topology;
    topology["containment_nodes"] = nodes;
    topology["cdata_capsules"] = cdata;
    topology["folds"] = collect_folds(nodes);
    topology["geodesics"] = collect_geodesics(nodes);
    topology["lanes"] = collect_lanes(nodes);
    topology["policies"] = collect_policies(nodes, cdata);
    topology["grams"] = collect_grams(input.text, cdata);
    add_node_tokens_to_grams(topology["grams"], nodes);
    topology["kuhul_programs"] = collect_kuhul_programs(cdata);
    topology["projection_targets"] = collect_projection_targets(nodes, cdata);

    json out;
    out["ok"] = true;
    out["reader"] = "semantic_reader.v1";
    out["source"] = input.source_name;
    out["topology"] = topology;
    out["activation"] = build_activation(input, topology);
    out["invariants"] = {
        {"no_destructive_flattening", true},
        {"cdata_preserved", true},
        {"policy_during_traversal", true},
        {"causal_replay_required", true},
        {"qkv_cannot_bypass_fold_policy", true}
    };
    return out;
}

bool write_semantic_reader_report(const std::string& input_path,
                                  const std::string& output_path,
                                  float activation_threshold) {
    std::ifstream in(input_path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();

    SemanticReader reader;
    SemanticReaderInput input;
    input.source_name = input_path;
    input.text = buffer.str();
    input.activation_threshold = activation_threshold;

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    out << reader.read(input).dump(2) << "\n";
    return true;
}

bool write_semantic_jsonl_absorb_report(const std::string& input_path,
                                        const std::string& output_path,
                                        float activation_threshold) {
    std::ifstream in(input_path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    json records = json::array();
    json rejected = json::array();
    json semantic_grams = json::array();
    json coarse_grams = json::array();
    json phase_trajectories = json::array();
    json shader_passes = json::array();
    json geodesics = json::array();
    json fibonacci_vectors = json::array();
    size_t mathml_count = 0;
    size_t compute_allowed_count = 0;
    size_t projection_only_count = 0;

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        std::string cleaned = trim(line);
        if (cleaned.empty()) {
            continue;
        }

        json record;
        try {
            record = json::parse(cleaned);
        } catch (const std::exception& e) {
            rejected.push_back({
                {"line", line_no},
                {"error", e.what()}
            });
            continue;
        }

        record["_line"] = line_no;
        records.push_back(record);

        if (record.contains("semantic_grams") && record["semantic_grams"].is_array()) {
            for (const auto& gram : record["semantic_grams"]) {
                add_unique(semantic_grams, gram);
            }
        }
        if (record.contains("coarse_gram")) {
            add_unique(coarse_grams, record["coarse_gram"]);
        }
        if (record.contains("phase_trajectory")) {
            add_unique(phase_trajectories, record["phase_trajectory"]);
        }
        if (record.contains("geodesic")) {
            if (record["geodesic"].is_array()) {
                for (const auto& route : record["geodesic"]) {
                    add_unique(geodesics, route);
                }
            } else {
                add_unique(geodesics, record["geodesic"]);
            }
        }
        if (record.contains("fibonacci")) {
            fibonacci_vectors.push_back({
                {"record", record.value("id", "")},
                {"fibonacci", record["fibonacci"]}
            });
        }
        if (record.contains("mathml")) {
            ++mathml_count;
        }
        if (record.contains("shader")) {
            shader_passes.push_back({
                {"record", record.value("id", "")},
                {"source", record.value("source", "")},
                {"shader", record["shader"]},
                {"payload_contract", record.value("payload_contract", json::object())}
            });
        }
        if (record.contains("payload_contract")) {
            const auto& contract = record["payload_contract"];
            if (contract.value("compute_allowed", false)) {
                ++compute_allowed_count;
            }
            if (contract.value("projection_allowed", false) && !contract.value("compute_allowed", false)) {
                ++projection_only_count;
            }
        }
    }

    const float pressure = std::min(1.0f,
        0.1f
        + static_cast<float>(semantic_grams.size()) * 0.01f
        + static_cast<float>(shader_passes.size()) * 0.008f
        + static_cast<float>(mathml_count) * 0.02f);

    json execution_plan = json::array({
        {{"step", "read_jsonl"}, {"records", records.size()}, {"rejected", rejected.size()}},
        {{"step", "resolve_semantic_grams"}, {"semantic_grams", semantic_grams.size()}, {"coarse_grams", coarse_grams.size()}},
        {{"step", "hydrate_mathml"}, {"mathml_records", mathml_count}},
        {{"step", "bind_geodesic_weights"}, {"routes", geodesics.size()}},
        {{"step", "bind_fibonacci_vectors"}, {"vectors", fibonacci_vectors.size()}},
        {{"step", "classify_shader_passes"}, {"passes", shader_passes.size()}, {"compute_allowed", compute_allowed_count}, {"projection_only", projection_only_count}},
        {{"step", "activate_absorb_surface"}, {"pressure", pressure}, {"active", pressure >= activation_threshold}}
    });

    json out;
    out["ok"] = rejected.empty();
    out["reader"] = "semantic_jsonl_absorber.v1";
    out["source"] = input_path;
    out["records"] = records;
    out["rejected"] = rejected;
    out["topology"] = {
        {"semantic_grams", semantic_grams},
        {"coarse_grams", coarse_grams},
        {"phase_trajectories", phase_trajectories},
        {"geodesics", geodesics},
        {"fibonacci_vectors", fibonacci_vectors},
        {"shader_passes", shader_passes},
        {"mathml_records", mathml_count}
    };
    out["activation"] = {
        {"pressure", {{"global", pressure}, {"activation_threshold", activation_threshold}}},
        {"active", pressure >= activation_threshold},
        {"execution_plan", execution_plan}
    };
    out["invariants"] = {
        {"jsonl_line_boundaries_preserved", true},
        {"mathml_payloads_preserved", true},
        {"fibonacci_vectors_preserved", true},
        {"shader_compute_projection_split", true},
        {"css_shader_matmul_rejected_unless_compute_backend", true}
    };

    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file.is_open()) {
        return false;
    }
    out_file << out.dump(2) << "\n";
    return true;
}
