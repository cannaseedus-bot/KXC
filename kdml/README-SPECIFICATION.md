# K'UHUL v0.1 Compiler Infrastructure — Frozen Reference Specification

**Status**: 📋 **Frozen Reference Specification**  
**Date**: 2026-07-12  
**Scope**: Layer 6 Compiler Infrastructure (Agent OS)  
**Version**: v0.1.0  

---

## 1. Overview

K'UHUL v0.1 is a **deterministic compiler infrastructure** that lowers semantic computation graphs into GPU-accelerated code. It bridges:

- **K'UHUL source** (`.kuhul` files with glyph notation)
- **IR representation** (in-memory computation DAGs)
- **DirectML backend** (GPU dispatch on Windows)
- **CPU fallback** (pure float arithmetic)

This specification defines the **frozen substrate contracts** that enable:
- Type-safe graph construction
- GPU-native lowering
- Deterministic execution
- Extensible operator catalog

---

## 2. Architecture

### 2.1 Pipeline Flow

```
Input (.kuhul)
    │
    ▼
Lexer (tokenization)
    │
    ▼
Parser (AST construction)
    │
    ▼
Semantic Analysis (type inference, phase assignment)
    │
    ▼
IR Builder (SSA form, graph construction)
    │
    ▼
Optimization Passes (const fold, fusion, dead code elimination)
    │
    ▼
DirectML Lowering (operator mapping, resource allocation)
    │
    ▼
GPU Dispatch (command list execution)
    │
    ▼
Output (results, metrics)
```

### 2.2 Key Layers

| Layer | Component | Purpose |
|-------|-----------|---------|
| **L6.1** | IR Type System | Value, Tensor, Node, Block, Function, Program |
| **L6.2** | DirectML Lowering | Graph validation, topological sort, operator mapping |
| **L6.3** | GPU Resource Manager | Buffer allocation, synchronization, profiling |
| **L6.4** | Operator Catalog | Conv2D, MatMul, ReLU, Pooling, etc. (extensible) |
| **L6.5** | Execution Context | Device management, performance tracking |

---

## 3. Core IR Type System

### 3.1 Value Type (Variant)

```cpp
using Value = std::variant<
    int64_t,                      // Int
    double,                       // Float
    bool,                         // Bool
    std::string,                  // String
    std::vector<Value>,          // List
    std::unordered_map<std::string, Value>, // Map
    IRNode*,                      // NodeRef
    IRFunction*,                  // CallRef
    std::monostate                // Void/Error
>;
```

**Semantics**:
- All computation values are represented as `Value` variants
- Type system is **dynamic but type-checkable**
- Maps to K'UHUL semantic glyphs

### 3.2 Tensor Type

```cpp
struct TensorType {
    std::vector<size_t> shape;     // Dimensions
    ValueType dtype;               // Element type (Int, Float)
    size_t size;                   // Total elements
    bool is_dynamic;               // Dynamic shape?
    std::optional<size_t> stride;  // Memory stride
    
    size_t total_elements() const {
        size_t total = 1;
        for (auto dim : shape) total *= dim;
        return total;
    }
    
    size_t bytes() const {
        size_t elem_size = (dtype == ValueType::Float) ? 4 : 8;
        return total_elements() * elem_size;
    }
};
```

**Invariants**:
- Shape must be non-empty
- All dimensions must be > 0
- Size = product of all dimensions

### 3.3 IR Node (Computation Unit)

```cpp
struct IRNode {
    // Identity
    std::string id;                              // Glyph name: @conv, @relu, etc.
    std::string op_code;                         // Canonical op: conv2d, relu, etc.
    size_t graph_id;                             // Unique ID in graph
    SourceLocation source_loc;                   // Debug: file, line, column
    
    // Data
    std::unordered_map<std::string, Value> attrs; // Parameters
    std::vector<IRNode*> inputs;                 // Input edges (non-owning)
    std::vector<IRNode*> outputs;                // Output edges (non-owning)
    ValueType output_type;                       // Output type
    TensorType output_tensor_shape;              // Output shape
    
    // Flags
    bool is_constant;              // Const folding candidate
    bool is_control_flow;          // If/loop/branch
    bool is_side_effecting;        // Has side effects
    bool is_pure;                  // No side effects
    bool is_fused;                 // Part of fusion group
    
    // Runtime
    mutable void* runtime_data;    // Backend-specific (e.g., GPU buffer ptr)
};
```

**Key Methods**:
- `has_attribute(key)` — Check if attribute exists
- `get_attribute<T>(key)` — Retrieve typed attribute
- `set_attribute(key, value)` — Set attribute
- `input_count()`, `output_count()` — Cardinality

**Semantics**:
- Node represents **one atomic operation**
- Inputs/outputs are **non-owning pointers** (managed by Block)
- Attributes encode **operation parameters** (strides, groups, etc.)
- Runtime data is **backend-specific** (GPU buffer, handle, etc.)

### 3.4 IR Edge

```cpp
struct IREdge {
    IRNode* source = nullptr;
    IRNode* target = nullptr;
    EdgeType type;                 // Forward, Data, Control, Loop, Bind, etc.
    size_t source_output_index;    // Which output of source
    size_t target_input_index;     // Which input of target
    std::optional<size_t> weight;  // Edge weight for optimization
    
    std::string name;              // Edge name for debug
    SourceLocation source_loc;     // Debug info
};
```

**Edge Types**:
- `Forward` — Standard dataflow
- `Data` — Explicit data dependency
- `Control` — Control dependency (no data)
- `Loop` — Backedge in loop
- `Bind` — Parameter binding
- `Tensor` — Tensor flow

### 3.5 IR Block (Graph Scope)

```cpp
struct IRBlock {
    std::string name;
    IRNode* entry_node;                          // Block root
    std::vector<std::unique_ptr<IRNode>> nodes;  // Owned nodes
    std::vector<IREdge> edges;                   // Internal edges
    std::vector<IRBlock*> subblocks;             // Nested blocks
    IRBlock* parent;                             // Parent block
    std::vector<IRNode*> execution_order;        // Topological order
    
    void add_node(std::unique_ptr<IRNode> node);
    IRNode* get_node(size_t id) const;
    void add_edge(const IREdge& edge);
    size_t node_count() const;
};
```

**Invariants**:
- `nodes` are **owned** (lifetime managed by Block)
- `edges` reference nodes by **non-owning pointers**
- Nodes added in order → `graph_id` = insertion index
- `execution_order` is populated by topological sort

### 3.6 IR Function

```cpp
struct IRFunction {
    std::string name;
    std::vector<std::string> params;             // Parameter names
    std::vector<ValueType> param_types;          // Parameter types
    IRBlock body;                                // Function body
    IRNode* return_node;                         // Return node
    
    std::unordered_map<std::string, Value> attrs; // Function attributes
    bool is_exported;                            // Exported to runtime
    bool is_inline;                              // Inline candidate
};
```

### 3.7 IR Program (Compilation Unit)

```cpp
struct IRProgram {
    std::string name;
    std::string version;
    
    std::vector<IRFunction> functions;
    std::vector<IRBlock> top_level_blocks;
    std::vector<IREdge> top_level_edges;
    
    std::unordered_map<std::string, Value> globals;      // Global constants
    std::unordered_map<std::string, ValueType> global_types;
};
```

---

## 4. DirectML Lowering Layer

### 4.1 Overview

The `DirectMLLowering` class converts IR graphs into DirectML operator chains:

```cpp
class DirectMLLowering {
public:
    // Initialize with D3D12/DirectML devices
    void Initialize(ID3D12Device*, IDMLDevice*, ID3D12CommandQueue*);
    
    // Main lowering functions
    IDMLCompiledOperator* LowerNodeToDML(IRNode* node);
    IDMLCompiledOperator* LowerGraph(IRGraph* graph);
    
    // Graph validation and transformation
    bool ValidateGraph(IRGraph* graph);
    std::vector<IRNode*> TopologicalSort(IRGraph* graph);
    
    // Resource management
    ID3D12Resource* AllocateResource(const TensorType& shape);
    void FreeResource(ID3D12Resource* resource);
};
```

### 4.2 Lowering Process

**Step 1: Validate Graph**
- Check for cycles (DAG requirement)
- Validate tensor shapes (no zero dimensions)
- Check all nodes have valid operators

**Step 2: Topological Sort**
- Order nodes for correct dependency resolution
- Use Kahn's algorithm (in-degree tracking)
- Results in valid execution order

**Step 3: Lower Each Node**
- For each node, call `BuildDMLDesc()`
- Map IR attributes to DirectML parameters
- Create `IDMLCompiledOperator`

**Step 4: Allocate Resources**
- For each node output, allocate D3D12 buffer
- Size = `tensor.bytes()`
- Flag: `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`

**Step 5: Build Execution Plan**
- Create D3D12 command list
- Create DML binding table
- Bind inputs/outputs for each operator
- Enqueue operators

### 4.3 Operator Mapping

**Supported Operators**:

| Op Code | DML Type | Parameters | Notes |
|---------|----------|------------|-------|
| `conv2d` | Conv2D | strides, dilations, groups, use_bias | 2D convolution |
| `matmul` | MatMul | transpose_a, transpose_b, alpha | Matrix multiply |
| `relu` | ReLU | None | ReLU activation |
| `softmax` | Softmax | axis | Softmax |
| `add` | Add | None | Element-wise add |
| `multiply` | Multiply | None | Element-wise mul |
| `pooling` | Pooling | kernel_size, strides, mode | Max or average |
| `activation` | Activation | type, alpha | ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU |
| `constant` | Constant | value | Constant tensor |

### 4.4 Example: Conv2D Lowering

**IR Node**:
```cpp
IRNode conv_node;
conv_node.op_code = "conv2d";
conv_node.set_attribute("strides", std::vector<Value>{2, 2});
conv_node.set_attribute("groups", 1);
conv_node.set_attribute("use_bias", true);
```

**DML Descriptor**:
```cpp
DMLOperatorDesc desc;
desc.type = DMLOperatorDesc::Type::Conv2D;
auto& params = desc.specific_params.emplace<DMLOperatorDesc::Conv2DParams>();
params.strides = {2, 2};
params.groups = 1;
params.use_bias = true;
```

**DirectML Call**:
```cpp
DML_CONVOLUTION_OPERATOR_DESC conv = {};
conv.InputTensor.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
conv.InputTensor.Sizes = {1, 3, 224, 224};
conv.OutputTensor.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
conv.OutputTensor.Sizes = {1, 32, 112, 112};
conv.Strides = {2, 2, 1, 1};
```

---

## 5. Usage Example

### 5.1 Build and Compile

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Run example
./build/examples/kuhul_conv_example

# Run tests
ctest --output-on-failure
```

### 5.2 Programmatic Usage

```cpp
#include "kuhul/ir_types.h"
#include "kuhul/lowering_to_dml.h"

using namespace KuhulIR;

// 1. Create IR program
IRProgram program;
program.name = "my_inference";

// 2. Create function
IRFunction fn;
fn.name = "forward";
fn.add_parameter("input", ValueType::Float);

// 3. Create nodes
auto conv_node = std::make_unique<IRNode>();
conv_node->op_code = "conv2d";
conv_node->set_attribute("strides", std::vector<Value>{1, 1});
fn.body.add_node(std::move(conv_node));

// 4. Add to program
program.add_function(std::move(fn));

// 5. Lower to DirectML
auto lowering = CreateDirectMLLowering(d3dDevice, dmlDevice, cmdQueue);
IRGraph graph;
auto compiled = lowering->LowerGraph(&graph);
```

---

## 6. Type Safety & Contracts

### 6.1 Invariants

**Value Variant**:
- All values are **type-checkable** via `std::holds_alternative<T>`
- Invalid type access returns `std::nullopt`

**Tensor Type**:
- Shape must be **non-empty** and **all dimensions > 0**
- Size = product of dimensions (enforced in `total_elements()`)
- Memory is **contiguous** (stride = element size)

**IR Node**:
- `inputs` and `outputs` are **non-owning pointers**
- Node lifetime managed by containing **IRBlock**
- Attributes are **dynamically typed** via `Value` variant

**IR Block**:
- Nodes are **owned** via `std::unique_ptr`
- `get_node(id)` is **safe** (bounds-checked)
- `execution_order` is computed lazily

**IR Edge**:
- Source and target pointers are **non-owning**
- Source and target **must not be null** (verified in `add_edge`)

### 6.2 Determinism

All operations are **deterministic**:
- Topological sort uses Kahn's algorithm (deterministic)
- Node iteration order is consistent (insertion order)
- DirectML execution is deterministic (no randomization)
- No external state or global variables

---

## 7. Extension Points

### 7.1 Adding New Operators

1. **Define operator in `DMLOperatorDesc::Type` enum**
   ```cpp
   enum class Type {
       Conv2D,
       MatMul,
       MyNewOp,  // Add here
       ...
   };
   ```

2. **Define parameters struct**
   ```cpp
   struct MyNewOpParams {
       double learning_rate;
       size_t batch_size;
   };
   ```

3. **Add to `SpecificParams` variant**
   ```cpp
   using SpecificParams = std::variant<
       Conv2DParams,
       MatMulParams,
       MyNewOpParams,  // Add here
       ...
   };
   ```

4. **Implement lowering in `BuildDMLDesc()`**
   ```cpp
   else if (node->op_code == "my_new_op") {
       desc.type = DMLOperatorDesc::Type::MyNewOp;
       auto& params = desc.specific_params.emplace<MyNewOpParams>();
       // ... extract from node->attrs
   }
   ```

5. **Implement DirectML operator creation in `CreateDMLOperator()`**
   ```cpp
   case DMLOperatorDesc::Type::MyNewOp: {
       // Create DML_MY_NEW_OP_OPERATOR_DESC
       // Compile with DXC
       break;
   }
   ```

### 7.2 Custom Nodes

Extend `IRNode` with domain-specific data:

```cpp
struct MyCustomNode : public IRNode {
    std::string domain;
    std::vector<float> custom_data;
    
    MyCustomNode() {
        op_code = "custom";
    }
};
```

---

## 8. Performance Considerations

### 8.1 Graph Optimization

- **Constant Folding**: Compute constant subgraphs at compile time
- **Dead Code Elimination**: Remove unused nodes
- **Operator Fusion**: Combine ops (Conv+ReLU → FusedConvReLU)
- **Memory Planning**: Pre-allocate buffers, minimize allocations

### 8.2 GPU Dispatch

- **Batch Operations**: Group small ops into single command list
- **Asynchronous Execution**: Overlap compute and data movement
- **Memory Coherency**: Minimize synchronization barriers

### 8.3 Memory Layout

- **Contiguous Tensors**: NCHW layout (row-major)
- **Stride Support**: Optional for sparse/strided tensors
- **Pre-allocation**: All buffers allocated before execution

---

## 9. Testing

### 9.1 Unit Tests

```bash
# Test IR types
ctest -R ir_types

# Test graph validation
ctest -R graph_validation

# Test lowering
ctest -R lowering
```

### 9.2 Integration Tests

```bash
# Run full example
./build/examples/kuhul_conv_example

# Expected output:
# - 3 nodes: Conv2D, ReLU, Pooling
# - Graph validation: passed
# - Topological sort: succeeded
# - All operators lowered to DirectML
```

---

## 10. File Structure

```
kuhul/
├── include/
│   └── kuhul/
│       ├── ir_types.h              # Type system (10.4KB)
│       ├── lowering_to_dml.h       # Lowering layer (2.8KB)
│       ├── lexer.h                 # (future)
│       ├── parser.h                # (future)
│       ├── optimizer.h             # (future)
│       └── executor.h              # (future)
│
├── src/
│   ├── lowering_to_dml.cpp         # Lowering impl (12.8KB)
│   ├── lexer.cpp                   # (future)
│   ├── parser.cpp                  # (future)
│   ├── optimizer.cpp               # (future)
│   └── executor.cpp                # (future)
│
├── tests/
│   └── test_ir_types.cpp           # Type tests (7.2KB)
│
├── examples/
│   ├── conv_example.cpp            # Conv2D example (9.8KB)
│   └── matmul_example.cpp          # (future)
│
├── CMakeLists.txt                  # Build config (2.4KB)
└── README.md                       # (this file)
```

---

## 11. Roadmap

| Phase | Component | Status | Notes |
|-------|-----------|--------|-------|
| **v0.1** | IR Types + DirectML | ✅ Complete | Frozen substrate |
| **v0.2** | Lexer + Parser | ⏳ Ready | ASCII glyph tokenization |
| **v0.3** | Semantic Analysis | ⏳ Ready | Type inference, K'UHUL phases |
| **v0.4** | Optimization Passes | ⏳ Ready | Const fold, fusion, DCE |
| **v0.5** | WebGPU Lowering | 📋 Planned | WGSL codegen |
| **v0.6** | CPU Fallback | 📋 Planned | Pure float arithmetic |
| **v0.7** | Profiling + Tracing | 📋 Planned | Per-op timing, memory |
| **v1.0** | Production Release | 📋 Planned | Full validation suite |

---

## 12. References

- **Binary IR Format**: See `BINARY-IR-SERIALIZATION-v1.0.md`
- **K'UHUL Fold Engine**: See `KUHUL-FOLD-ENGINE-v0.1.md`
- **MATRIX DAG Runtime**: See `MATRIX-DAG-RUNTIME-v0.1.md`
- **Layer 5 Firewall**: See `layer5-firewall.hpp`
- **Production Release**: See `PRODUCTION-RELEASE-PACKAGE-v1.7.md`

---

**Status**: ✅ **Frozen Reference Specification v0.1**  
**Ready for**: Agent OS integration, Layer 6 compiler implementation  
**Next Steps**: Lexer + Parser implementation, semantic analysis bridge
