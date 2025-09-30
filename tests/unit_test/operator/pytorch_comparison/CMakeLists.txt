# PyTorch对比示例 CMakeLists.txt
# 用于编译OriginDL对比示例

cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(pytorch_comparison)

# 继承主项目的编译选项
message(STATUS "pytorch_comparison inheriting compilation options from main project")

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 设置包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd/googletest/googletest/include)

# 设置输出目录
set(COMPARISON_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/pytorch_comparison)

# 添加OriginDL Sum操作示例
add_executable(origindl_sum_example origindl_sum_example.cpp)
add_dependencies(origindl_sum_example origindl)
target_link_libraries(origindl_sum_example origindl pthread)
target_include_directories(origindl_sum_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(origindl_sum_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${COMPARISON_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${COMPARISON_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${COMPARISON_OUTPUT_DIR}
)

# 添加OriginDL Transpose操作示例
add_executable(origindl_transpose_example origindl_transpose_example.cpp)
add_dependencies(origindl_transpose_example origindl)
target_link_libraries(origindl_transpose_example origindl pthread)
target_include_directories(origindl_transpose_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(origindl_transpose_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${COMPARISON_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${COMPARISON_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${COMPARISON_OUTPUT_DIR}
)

# 添加OriginDL MatMul操作示例
add_executable(origindl_matmul_example origindl_matmul_example.cpp)
add_dependencies(origindl_matmul_example origindl)
target_link_libraries(origindl_matmul_example origindl pthread)
target_include_directories(origindl_matmul_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(origindl_matmul_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${COMPARISON_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${COMPARISON_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${COMPARISON_OUTPUT_DIR}
)

# 添加OriginDL Reshape操作示例
add_executable(origindl_reshape_example origindl_reshape_example.cpp)
add_dependencies(origindl_reshape_example origindl)
target_link_libraries(origindl_reshape_example origindl pthread)
target_include_directories(origindl_reshape_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(origindl_reshape_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${COMPARISON_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${COMPARISON_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${COMPARISON_OUTPUT_DIR}
)

# 添加自定义目标来编译所有对比示例
add_custom_target(compile_comparison_examples
    DEPENDS origindl_sum_example origindl_transpose_example origindl_matmul_example origindl_reshape_example
    COMMENT "Compile all PyTorch comparison examples"
)

# 添加自定义目标来运行所有对比示例
add_custom_target(run_comparison_examples
    COMMAND ${CMAKE_SOURCE_DIR}/tests/unit_test/operator/pytorch_comparison/run_comparison.sh
    DEPENDS origindl_sum_example origindl_transpose_example origindl_matmul_example origindl_reshape_example
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/unit_test/operator/pytorch_comparison
    COMMENT "Run all PyTorch comparison examples"
)

# 打印配置信息
message(STATUS "PyTorch comparison examples configuration completed")
message(STATUS "Comparison examples output directory: ${COMPARISON_OUTPUT_DIR}")
