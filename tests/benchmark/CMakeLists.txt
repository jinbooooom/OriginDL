cmake_minimum_required(VERSION 3.10)

# 设置包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 设置可执行文件输出目录
set(BENCHMARK_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/benchmark)

# Add算子性能测试
add_executable(bench_add add/bench_add.cpp common/benchmark_framework.cpp)
add_dependencies(bench_add origindl)
target_link_libraries(bench_add 
    origindl
    pthread
)

# Origin后端：如果启用CUDA，添加CUDA相关链接
if(ENABLE_CUDA)
    target_link_libraries(bench_add originmat_cuda ${CUDA_LIBRARIES})
endif()

target_include_directories(bench_add PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_target_properties(bench_add PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BENCHMARK_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BENCHMARK_OUTPUT_DIR}
)

# MatMul算子性能测试
add_executable(bench_matmul matmul/bench_matmul.cpp common/benchmark_framework.cpp)
add_dependencies(bench_matmul origindl)
target_link_libraries(bench_matmul 
    origindl
    pthread
)

# Origin后端：如果启用CUDA，添加CUDA相关链接
if(ENABLE_CUDA)
    target_link_libraries(bench_matmul originmat_cuda ${CUDA_LIBRARIES})
endif()

target_include_directories(bench_matmul PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_target_properties(bench_matmul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BENCHMARK_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BENCHMARK_OUTPUT_DIR}
)

# Conv2d算子性能测试
add_executable(bench_conv2d conv2d/bench_conv2d.cpp common/benchmark_framework.cpp)
add_dependencies(bench_conv2d origindl)
target_link_libraries(bench_conv2d 
    origindl
    pthread
)

# Origin后端：如果启用CUDA，添加CUDA相关链接
if(ENABLE_CUDA)
    target_link_libraries(bench_conv2d originmat_cuda ${CUDA_LIBRARIES})
endif()

target_include_directories(bench_conv2d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_target_properties(bench_conv2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BENCHMARK_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BENCHMARK_OUTPUT_DIR}
)

message(STATUS "Benchmark tests configuration completed")
