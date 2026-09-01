// lowering_to_dml.h - DirectML Lowering Layer for K'UHUL IR
// Frozen Reference Implementation for Layer 6 Compiler
// Date: 2026-07-12

#pragma once
#include "ir_types.h"
#include <memory>
#include <unordered_map>
#include <vector>

// Forward declarations for D3D12/DirectML
struct ID3D12Device;
struct IDMLDevice;
struct ID3D12CommandQueue;
struct IDMLCompiledOperator;
struct ID3D12Resource;

namespace KuhulIR {

class DirectMLLowering {
public:
    DirectMLLowering();
    ~DirectMLLowering();
    
    // Initialize with D3D12/DirectML devices
    void Initialize(ID3D12Device* d3dDevice, IDMLDevice* dmlDevice, ID3D12CommandQueue* cmdQueue);
    
    // ===== Main Lowering Functions =====
    IDMLCompiledOperator* LowerNodeToDML(IRNode* node);
    IDMLCompiledOperator* LowerGraph(IRGraph* graph);
    
    // Build DML operator descriptor from IR node
    DMLOperatorDesc BuildDMLDesc(IRNode* node);
    
    // Create DirectML operator from descriptor
    IDMLCompiledOperator* CreateDMLOperator(const DMLOperatorDesc& desc);
    
    // ===== Graph Validation =====
    bool ValidateGraph(IRGraph* graph);
    bool HasCycle(IRGraph* graph);
    bool HasCycleDFS(IRNode* node,
                     std::unordered_map<IRNode*, bool>& visited,
                     std::unordered_map<IRNode*, bool>& recursion_stack);
    bool ValidateTensorShapes(IRNode* node);
    
    // ===== Topological Sort =====
    std::vector<IRNode*> TopologicalSort(IRGraph* graph);
    
    // ===== Resource Management =====
    ID3D12Resource* AllocateResource(const TensorType& shape);
    void FreeResource(ID3D12Resource* resource);
    
    // ===== Execution Planning =====
    IDMLCompiledOperator* BuildExecutionPlan(
        const std::vector<IRNode*>& sorted_nodes,
        const std::unordered_map<IRNode*, IDMLCompiledOperator*>& node_to_dml,
        const std::unordered_map<IRNode*, ID3D12Resource*>& node_to_resource);
    
    // ===== Shape Inference =====
    std::vector<size_t> GetInputShapes(IRNode* node);
    std::vector<size_t> GetOutputShapes(IRNode* node);
    std::vector<size_t> ExtractSizeVector(const std::vector<Value>& values);
    
    // ===== Helper Functions =====
    int MapToDMLType(DMLOperatorDesc::Type type);
    
    void LogError(const std::string& message);
    void LogInfo(const std::string& message);
    
private:
    // D3D12/DirectML devices
    ID3D12Device* m_d3dDevice = nullptr;
    IDMLDevice* m_dmlDevice = nullptr;
    ID3D12CommandQueue* m_cmdQueue = nullptr;
    
    // Resource tracking
    std::vector<ID3D12Resource*> m_allocated_resources;
};

// Factory function
std::unique_ptr<DirectMLLowering> CreateDirectMLLowering(
    ID3D12Device* d3dDevice,
    IDMLDevice* dmlDevice,
    ID3D12CommandQueue* cmdQueue);

} // namespace KuhulIR
