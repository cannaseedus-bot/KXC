// Fold2DCompiler.h — K'UHUL 2D Fold Pressure Compiler
//
// Parses a KXML graph, computes the fold×rank pressure table,
// generates the ballooning matrix, compiles each node to HLSL,
// and emits KuhulDispatchTable.h + SafeTensor.h.
//
// Fold pressure table (pressure[fold][rank 0..4]):
//   COMPUTE   8.0  4.0  2.0  1.0  0.5    high  — tight GPU packing
//   STORAGE   4.0  2.0  1.0  0.5  0.25   med   — balanced
//   META      1.0  0.5  0.25 ...         low   — semantic expansion
//   ROUTING   2.0  1.0  1.0  2.0  4.0    var   — fan-out dependent
//   UI        0.25 0.125 0.0625 ...      min   — human-readable
//
// Ballooning: balloon_factor = source_pressure / target_pressure
//   COMPUTE rank-2 (2.0) → UI rank-2 (0.0625) = 32× expansion
//   These expansions explain why [dbg] log lines appear "larger" —
//   they balloon from COMPUTE_FOLD to UI_FOLD when emitted.
//
// SafeTensor: gravity bound = 20.0 / pressure
//   COMPUTE rank-2 (2.0) → bound = ±10.0
//   UI      rank-2 (0.0625) → bound = ±320.0  (deliberately loose)
//   The sloppy iGPU run had COMPUTE tensors with UI-level bounds (no pressure).
//
// Pressure reserves (from KuhulPhysics.h):
//   Each fold node has a PressureReserve that absorbs shocks before live
//   gravity constraints tighten. Full reserves → logit_bound stays at 20.
//   Empty reserves → KuhulPhysicsSolver Rule 1 fires → logit_bound → 18.

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <iostream>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>

using namespace DirectX;
using namespace DirectX::PackedVector;

// ─── Fold types ───────────────────────────────────────────────────────────────

enum class Fold : uint8_t {
    COMPUTE = 0, STORAGE = 1, META = 2, ROUTING = 3, UI = 4
};

// ─── PressureLevel ────────────────────────────────────────────────────────────

struct PressureLevel {
    float compute[5] = {8.0f, 4.0f,    2.0f,     1.0f,      0.5f    };
    float storage[5] = {4.0f, 2.0f,    1.0f,     0.5f,      0.25f   };
    float meta   [5] = {1.0f, 0.5f,    0.25f,    0.125f,    0.0625f };
    float routing[5] = {2.0f, 1.0f,    1.0f,     2.0f,      4.0f    };
    float ui     [5] = {0.25f,0.125f,  0.0625f,  0.03125f,  0.015625f};

    float Get(Fold f, size_t rank) const {
        rank = std::min(rank, size_t(4));
        switch (f) {
            case Fold::COMPUTE: return compute[rank];
            case Fold::STORAGE: return storage[rank];
            case Fold::META:    return meta[rank];
            case Fold::ROUTING: return routing[rank];
            case Fold::UI:      return ui[rank];
        }
        return 1.0f;
    }

    float BalloonFactor(Fold src, Fold dst, size_t rank = 2) const {
        return Get(src, rank) / Get(dst, rank);
    }
};

// ─── KXMLNode / KXMLEdge ──────────────────────────────────────────────────────

struct KXMLNode {
    std::string id, device, phase;
    Fold fold = Fold::COMPUTE;
    bool antigravity = false;
    std::vector<std::string> ops;
};

struct KXMLEdge {
    std::string from, to;
    bool bidirectional = false;
    float gradientScale = 0.001f;
};

// ─── SafeTensor ───────────────────────────────────────────────────────────────

class SafeTensor {
    float*  m_data;
    size_t  m_size;
    float   m_pressure;
    float   m_bound;      // = 20.0 / pressure

public:
    SafeTensor(size_t sz, float pressure)
        : m_size(sz), m_pressure(pressure), m_bound(20.0f / std::max(pressure, 0.001f)) {
        m_data = new float[sz]();
    }
    ~SafeTensor() { delete[] m_data; }

    void ApplyGravity() {
        for (size_t i = 0; i < m_size; i++)
            m_data[i] = std::min(std::max(m_data[i], -m_bound), m_bound);
    }

    XMVECTOR Load4(size_t off) const {
        return XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(m_data + off));
    }
    void Store4(size_t off, XMVECTOR v) {
        XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(m_data + off), v);
        ApplyGravity();
    }

    float  pressure() const { return m_pressure; }
    float  bound()    const { return m_bound; }
    size_t size()     const { return m_size; }
};

// ─── Fold2DCompiler ───────────────────────────────────────────────────────────

class Fold2DCompiler {
    PressureLevel m_P;
    std::unordered_map<std::string, KXMLNode>  m_nodes;
    std::vector<KXMLEdge>                       m_edges;
    std::vector<std::string>                    m_kernels;

    static std::string FoldStr(Fold f) {
        switch (f) {
            case Fold::COMPUTE: return "COMPUTE";  case Fold::STORAGE: return "STORAGE";
            case Fold::META:    return "META";      case Fold::ROUTING: return "ROUTING";
            case Fold::UI:      return "UI";
        }
        return "UNKNOWN";
    }

    static Fold ParseFold(const std::string& s) {
        if (s.find("COMPUTE") != std::string::npos) return Fold::COMPUTE;
        if (s.find("STORAGE") != std::string::npos) return Fold::STORAGE;
        if (s.find("META")    != std::string::npos) return Fold::META;
        if (s.find("ROUTING") != std::string::npos) return Fold::ROUTING;
        return Fold::UI;
    }

public:
    bool ParseKXML(const std::string& path) {
        std::ifstream f(path); if (!f) return false;
        std::string line;
        std::regex nodeRe(R"(<node\s+id="([^"]+)"\s+fold="([^"]+)"\s+device="([^"]+)"\s+phase="([^"]+)")");
        std::regex edgeRe(R"(<edge\s+from="([^"]+)"\s+to="([^"]+))");
        std::regex antiRe(R"(antigravity="true")");

        KXMLNode* cur = nullptr;
        while (std::getline(f, line)) {
            std::smatch m;
            if (std::regex_search(line, m, nodeRe)) {
                KXMLNode n; n.id=m[1]; n.fold=ParseFold(m[2]); n.device=m[3]; n.phase=m[4];
                n.antigravity = std::regex_search(line, antiRe);
                m_nodes[n.id] = n; cur = &m_nodes[n.id];
            } else if (std::regex_search(line, m, edgeRe)) {
                KXMLEdge e; e.from=m[1]; e.to=m[2];
                e.bidirectional = (line.find("bidirectional=\"true\"") != std::string::npos);
                m_edges.push_back(e);
            } else if (cur && line.find("<op ") != std::string::npos) {
                cur->ops.push_back(line);
            }
        }
        std::cout << "[Fold2DCompiler] parsed " << m_nodes.size() << " nodes, "
                  << m_edges.size() << " edges\n";
        return true;
    }

    void PrintPressureTable() {
        std::cout << "\n[Fold2DCompiler] Pressure Table\n";
        printf("%-9s %8s %8s %8s %8s %8s\n", "Fold","0D","1D","2D","3D","4D");
        for (auto& [name,fold] : std::vector<std::pair<std::string,Fold>>{
             {"COMPUTE",Fold::COMPUTE},{"STORAGE",Fold::STORAGE},{"META",Fold::META},
             {"ROUTING",Fold::ROUTING},{"UI",Fold::UI}}) {
            printf("%-9s", name.c_str());
            for (int r=0;r<5;r++) printf(" %7.4f", m_P.Get(fold,r));
            printf("\n");
        }

        std::cout << "\n[Fold2DCompiler] Ballooning Matrix (rank-2, source→target)\n";
        std::vector<std::pair<std::string,Fold>> folds = {
            {"COMPUTE",Fold::COMPUTE},{"STORAGE",Fold::STORAGE},{"META",Fold::META},
            {"ROUTING",Fold::ROUTING},{"UI",Fold::UI}};
        printf("%-9s", "");
        for (auto&[n,_]:folds) printf(" %8s",n.c_str());
        printf("\n");
        for (auto&[sn,sf]:folds) {
            printf("%-9s",sn.c_str());
            for (auto&[tn,tf]:folds) printf(" %7.1fx", m_P.BalloonFactor(sf,tf,2));
            printf("\n");
        }
    }

    std::string GenerateKernel(const KXMLNode& n) {
        std::ostringstream ss;
        ss << "// K'UHUL generated kernel: " << n.id << "  fold=" << FoldStr(n.fold)
           << "  device=" << n.device << "  phase=" << n.phase << "\n";
        // Thread group: proportional to fold pressure
        int tg = (n.fold==Fold::COMPUTE)?256:(n.fold==Fold::STORAGE)?64:
                 (n.fold==Fold::META)?32:1;
        ss << "[numthreads(" << tg << ", 1, 1)]\n";
        ss << "void CSMain(uint3 id : SV_DispatchThreadID) {\n";
        float bound = 20.0f / std::max(m_P.Get(n.fold,2), 0.001f);
        ss << "  float bound = " << bound << "f;\n";
        if (n.antigravity) {
            ss << "  // ANTIGRAVITY node — observe only, no gravity bounds applied\n";
        }
        for (auto& op : n.ops) {
            if (op.find("fibonacci") != std::string::npos)
                ss << "  output[id.x] = clamp(FibFastDouble(input[id.x]), -bound, bound);\n";
            else if (op.find("gemm") != std::string::npos)
                ss << "  output[id.x] = clamp(dot(input4[id.x], weight4[id.x]), -bound, bound);\n";
            else
                ss << "  output[id.x] = clamp(input[id.x], -bound, bound);  // " << op << "\n";
        }
        ss << "}\n";
        return ss.str();
    }

    void GenerateDispatchTable(const std::string& path) {
        std::ofstream f(path);
        f << "// AUTO-GENERATED by Fold2DCompiler — DO NOT EDIT\n";
        f << "#pragma once\n#include <unordered_map>\n#include <string>\n\n";
        f << "struct MappingEntry { std::string fold; float pressure; "
             "std::string device; std::string kernel; float balloonToUI; bool antigravity; };\n\n";
        f << "static const std::unordered_map<std::string,MappingEntry> kuhulDispatch = {\n";
        for (auto& [id,n] : m_nodes) {
            float p = m_P.Get(n.fold,2);
            float b = m_P.BalloonFactor(n.fold, Fold::UI, 2);
            f << "  {\"" << id << "\",{\"" << FoldStr(n.fold) << "\"," << p << "f,\""
              << n.device << "\",\"" << id << "_kernel\"," << b << "f,"
              << (n.antigravity?"true":"false") << "}},\n";
        }
        f << "};\n";
        std::cout << "[Fold2DCompiler] → " << path << "\n";
    }

    void GenerateSafeTensorHeader(const std::string& path) {
        std::ofstream f(path);
        f << R"(// SafeTensor.h — AUTO-GENERATED by Fold2DCompiler
// gravity_bound = 20.0 / fold_pressure
// Ensures tensors never exceed physics bounds regardless of fold they're in.
#pragma once
#include <DirectXMath.h>
#include <algorithm>
using namespace DirectX;

class SafeTensor {
    float* m_data; size_t m_sz; float m_bound;
public:
    SafeTensor(size_t sz, float pressure)
        : m_sz(sz), m_data(new float[sz]()), m_bound(20.0f / std::max(pressure,0.001f)) {}
    ~SafeTensor() { delete[] m_data; }
    void ApplyGravity() {
        for (size_t i=0;i<m_sz;i++) m_data[i]=std::min(std::max(m_data[i],-m_bound),m_bound);
    }
    XMVECTOR Load4(size_t off) const { return XMLoadFloat4((XMFLOAT4*)(m_data+off)); }
    void Store4(size_t off, XMVECTOR v) { XMStoreFloat4((XMFLOAT4*)(m_data+off),v); ApplyGravity(); }
    float bound() const { return m_bound; }
    SafeTensor* BalloonTo(float targetPressure) const {
        float factor = m_bound * targetPressure / 20.0f;  // inverse pressure ratio
        return new SafeTensor(m_sz, targetPressure);
    }
};
)";
        std::cout << "[Fold2DCompiler] → " << path << "\n";
    }

    void Compile(const std::string& kxmlPath, const std::string& outDir = ".") {
        std::cout << "\n[Fold2DCompiler] K'UHUL 2D Compiler\n";
        if (!ParseKXML(kxmlPath)) { std::cerr << "parse failed\n"; return; }
        PrintPressureTable();
        for (auto& [id,n] : m_nodes) {
            std::string src = GenerateKernel(n);
            std::ofstream f(outDir + "/" + id + "_kernel.hlsl");
            f << src; f.close();
            m_kernels.push_back(id);
        }
        GenerateDispatchTable(outDir + "/KuhulDispatchTable.h");
        GenerateSafeTensorHeader(outDir + "/SafeTensor.h");
        std::cout << "[Fold2DCompiler] Done. Nodes=" << m_nodes.size()
                  << " Kernels=" << m_kernels.size() << " Edges=" << m_edges.size() << "\n";
    }
};
