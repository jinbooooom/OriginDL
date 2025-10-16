# CUDA单元测试 CMakeLists.txt
# 独立的CUDA测试配置，使用标准Tensor API测试CUDA功能
# 文件组织结构与CPU测试保持一致：tensor/ operator/ autograd/

cmake_minimum_required(VERSION 3.18)

# 设置项目名称
project(unit_tests_cuda)

# 检查CUDA支持
if(NOT ENABLE_CUDA)
    message(STATUS "CUDA support is disabled, skipping CUDA unit tests")
    return()
endif()

# 检查CUDA库是否存在
if(NOT TARGET originmat_cuda)
    message(STATUS "originmat_cuda library not found, skipping CUDA unit tests")
    return()
endif()

# 启用CUDA语言
enable_language(CUDA)
# 继承主项目的C++标准，不强制设置CUDA标准
# set(CMAKE_CUDA_STANDARD 11)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA库
find_package(CUDAToolkit REQUIRED)

# 继承主项目的编译选项
message(STATUS "CUDA unit tests inheriting compilation options from main project")

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 使用主项目编译的GoogleTest库
message(STATUS "Using GoogleTest libraries compiled by main project")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 设置包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${GTEST_INCLUDE_DIRS})

# 设置CUDA测试输出目录
set(CUDA_UNIT_TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/unit_test_cuda)
message(STATUS "CUDA unit test output directory: ${CUDA_UNIT_TEST_OUTPUT_DIR}")

# ============================================================================
# CUDA张量测试 - 测试张量创建、转换、属性等基础功能
# ============================================================================

# CUDA张量基础功能测试
add_executable(test_cuda_tensor tensor/test_cuda_tensor.cpp)
add_dependencies(test_cuda_tensor gtest gtest_main origindl originmat_cuda)
target_link_libraries(test_cuda_tensor 
    ${GTEST_LIBRARIES}
    origindl
    originmat_cuda
    CUDA::cudart
    pthread
)
target_include_directories(test_cuda_tensor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)
set_target_properties(test_cuda_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUDA_UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CUDA_UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CUDA_UNIT_TEST_OUTPUT_DIR}
)

# ============================================================================
# CUDA算子测试 - 测试各种运算操作，每个算子一个测试文件
# ============================================================================

# CUDA加法算子测试
add_executable(test_cuda_add operator/test_cuda_add.cpp)
add_dependencies(test_cuda_add gtest gtest_main origindl originmat_cuda)
target_link_libraries(test_cuda_add 
    ${GTEST_LIBRARIES}
    origindl
    originmat_cuda
    CUDA::cudart
    pthread
)
target_include_directories(test_cuda_add PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)
set_target_properties(test_cuda_add PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUDA_UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CUDA_UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CUDA_UNIT_TEST_OUTPUT_DIR}
)

# CUDA减法算子测试
add_executable(test_cuda_sub operator/test_cuda_sub.cpp)
add_dependencies(test_cuda_sub gtest gtest_main origindl originmat_cuda)
target_link_libraries(test_cuda_sub 
    ${GTEST_LIBRARIES}
    origindl
    originmat_cuda
    CUDA::cudart
    pthread
)
target_include_directories(test_cuda_sub PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)
set_target_properties(test_cuda_sub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUDA_UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CUDA_UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CUDA_UNIT_TEST_OUTPUT_DIR}
)

# CUDA乘法算子测试
add_executable(test_cuda_mul operator/test_cuda_mul.cpp)
add_dependencies(test_cuda_mul gtest gtest_main origindl originmat_cuda)
target_link_libraries(test_cuda_mul 
    ${GTEST_LIBRARIES}
    origindl
    originmat_cuda
    CUDA::cudart
    pthread
)
target_include_directories(test_cuda_mul PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)
set_target_properties(test_cuda_mul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUDA_UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CUDA_UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CUDA_UNIT_TEST_OUTPUT_DIR}
)

# 其他算子测试可以在这里添加
# add_executable(test_cuda_div operator/test_cuda_div.cpp)
# add_executable(test_cuda_exp operator/test_cuda_exp.cpp)
# add_executable(test_cuda_log operator/test_cuda_log.cpp)
# add_executable(test_cuda_sqrt operator/test_cuda_sqrt.cpp)
# add_executable(test_cuda_square operator/test_cuda_square.cpp)
# add_executable(test_cuda_negate operator/test_cuda_negate.cpp)

# ============================================================================
# CUDA自动微分测试 - 为未来自动微分功能预留
# ============================================================================

# 暂时没有自动微分测试，等实现后再添加
# add_executable(test_cuda_autograd autograd/test_cuda_autograd.cpp)

# ============================================================================
# 启用测试
# ============================================================================

enable_testing()

# 添加CUDA测试
add_test(NAME cuda_tensor_test COMMAND ${CUDA_UNIT_TEST_OUTPUT_DIR}/test_cuda_tensor)
add_test(NAME cuda_add_test COMMAND ${CUDA_UNIT_TEST_OUTPUT_DIR}/test_cuda_add)
add_test(NAME cuda_sub_test COMMAND ${CUDA_UNIT_TEST_OUTPUT_DIR}/test_cuda_sub)
add_test(NAME cuda_mul_test COMMAND ${CUDA_UNIT_TEST_OUTPUT_DIR}/test_cuda_mul)

# 设置测试属性
set_tests_properties(cuda_tensor_test PROPERTIES TIMEOUT 60)
set_tests_properties(cuda_add_test PROPERTIES TIMEOUT 60)
set_tests_properties(cuda_sub_test PROPERTIES TIMEOUT 60)
set_tests_properties(cuda_mul_test PROPERTIES TIMEOUT 60)

# 打印配置信息
message(STATUS "CUDA unit tests configuration completed")
message(STATUS "CUDA unit test output directory: ${CUDA_UNIT_TEST_OUTPUT_DIR}")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 添加自定义目标来运行所有CUDA测试
add_custom_target(run_cuda_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_cuda_tensor test_cuda_add test_cuda_sub test_cuda_mul
    COMMENT "Run all CUDA unit tests"
)

# 添加自定义目标来编译和运行CUDA测试
add_custom_target(test_cuda_all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_cuda_tensor test_cuda_add test_cuda_sub test_cuda_mul
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_cuda_tensor test_cuda_add test_cuda_sub test_cuda_mul
    COMMENT "Compile and run all CUDA unit tests"
)