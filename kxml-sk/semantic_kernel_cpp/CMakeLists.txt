cmake_minimum_required(VERSION 3.10)
project(semantic_kernel_cli)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Main Library Source
set(LIB_SOURCES
    src/semantic_kernel.cpp
    src/compiler.cpp
    src/planner.cpp
    src/memory.cpp
    src/dx12_executor.cpp
    src/executor.cpp
    src/batch_executor.cpp
    src/runtime_loop.cpp
    src/scheduler.cpp
    src/tokenizer.cpp
    src/quant.cpp
    src/simd_kernels.cpp
    src/protobuf_writer.cpp
    src/metric_normalization.cpp
    src/geodesic_flow_layer.cpp
    src/dx12_device_factory.cpp
    src/semantic_reader.cpp
)



# Headers (for IDE integration)
set(LIB_HEADERS
    include/semantic_kernel.h
    include/compiler.h
    include/planner.h
    include/memory.h
    include/dx12_executor.h
    include/executor.h
    include/ir.h
    include/knobs.h
    include/kv_cache.h
    include/quant.h
    include/runtime_loop.h
    include/scheduler.h
    include/simd_kernels.h
    include/dds_shard_loader.h
    include/evolution_bot.h
    include/replay_lane_manager.h
    include/field_execution_engine.h
    include/legality_verifier.h
    include/projection_expert.h
    include/memory_compressor.h
    include/dx12_device_factory.h
    include/metric_normalization.h
    include/geodesic_flow_layer.h
    include/shared_memory_bridge.h
    include/semantic_reader.h
)




# Shaders
set(SHADERS
    src/adapter_fusion.hlsl
    src/bimodal_attention.hlsl
    src/geodesic_flow.hlsl
)

add_library(semantic_kernel_lib STATIC ${LIB_SOURCES} ${LIB_HEADERS})

# Link D3D12 for Windows
if(WIN32)
    target_link_libraries(semantic_kernel_lib d3d12 dxgi)
endif()

# CLI Target
add_executable(semantic_kernel_cli src/main.cpp)
target_link_libraries(semantic_kernel_cli semantic_kernel_lib)

# Verification Target
add_executable(verify_asx verify_asx_runtime.cpp)
target_link_libraries(verify_asx semantic_kernel_lib)

# Robustness Tests (Phase 5)
add_executable(test_compressor test_compressor.cpp)
target_link_libraries(test_compressor semantic_kernel_lib)

add_executable(unit_tests unit_tests.cpp)
target_link_libraries(unit_tests semantic_kernel_lib)



# Install rules (Optional)
install(TARGETS semantic_kernel_cli verify_asx RUNTIME DESTINATION bin)
