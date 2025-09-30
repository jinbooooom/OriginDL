# Tensor单元测试 CMakeLists.txt
# 参考tests/3rd/googleTest/CMakeLists.txt的配置方式

cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(unit_tests)

# 继承主项目的编译选项
message(STATUS "unit_tests inheriting compilation options from main project")

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 使用主项目编译的GoogleTest库
message(STATUS "Using GoogleTest libraries compiled by main project")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 设置包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${GTEST_INCLUDE_DIRS})

# 设置单元测试输出目录
set(UNIT_TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/unit_test)

# 添加Tensor基础测试
add_executable(test_tensor_create tensor/test_create_tensor.cpp)
add_dependencies(test_tensor_create gtest gtest_main origindl)
target_link_libraries(test_tensor_create 
    ${GTEST_LIBRARIES}
    origindl
    pthread
)
target_include_directories(test_tensor_create PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)
set_target_properties(test_tensor_create PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 添加算子测试
# 基础算术算子测试
add_executable(test_add operator/test_add.cpp)
add_dependencies(test_add gtest gtest_main origindl)
target_link_libraries(test_add ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_add PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_add PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_sub operator/test_sub.cpp)
add_dependencies(test_sub gtest gtest_main origindl)
target_link_libraries(test_sub ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_sub PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_sub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_mul operator/test_mul.cpp)
add_dependencies(test_mul gtest gtest_main origindl)
target_link_libraries(test_mul ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_mul PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_mul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_div operator/test_div.cpp)
add_dependencies(test_div gtest gtest_main origindl)
target_link_libraries(test_div ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_div PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_div PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 一元算子测试
add_executable(test_neg operator/test_neg.cpp)
add_dependencies(test_neg gtest gtest_main origindl)
target_link_libraries(test_neg ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_neg PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_neg PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_square operator/test_square.cpp)
add_dependencies(test_square gtest gtest_main origindl)
target_link_libraries(test_square ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_square PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_square PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_exp operator/test_exp.cpp)
add_dependencies(test_exp gtest gtest_main origindl)
target_link_libraries(test_exp ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_exp PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_exp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_pow operator/test_pow.cpp)
add_dependencies(test_pow gtest gtest_main origindl)
target_link_libraries(test_pow ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_pow PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_pow PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 聚合算子测试
add_executable(test_sum operator/test_sum.cpp)
add_dependencies(test_sum gtest gtest_main origindl)
target_link_libraries(test_sum ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_sum PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_sum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_sum_to operator/test_sum_to.cpp)
add_dependencies(test_sum_to gtest gtest_main origindl)
target_link_libraries(test_sum_to ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_sum_to PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_sum_to PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_broadcast_to operator/test_broadcast_to.cpp)
add_dependencies(test_broadcast_to gtest gtest_main origindl)
target_link_libraries(test_broadcast_to ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_broadcast_to PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_broadcast_to PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 形状操作算子测试
add_executable(test_transpose operator/test_transpose.cpp)
add_dependencies(test_transpose gtest gtest_main origindl)
target_link_libraries(test_transpose ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_transpose PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_transpose PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_reshape operator/test_reshape.cpp)
add_dependencies(test_reshape gtest gtest_main origindl)
target_link_libraries(test_reshape ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_reshape PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_reshape PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 矩阵运算算子测试
add_executable(test_matmul operator/test_matmul.cpp)
add_dependencies(test_matmul gtest gtest_main origindl)
target_link_libraries(test_matmul ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_matmul PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_matmul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 自动微分测试
add_executable(test_linear_regression autograd/test_linear_regression.cpp)
add_dependencies(test_linear_regression gtest gtest_main origindl)
target_link_libraries(test_linear_regression ${GTEST_LIBRARIES} origindl pthread)
target_include_directories(test_linear_regression PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
set_target_properties(test_linear_regression PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)


# 启用测试
enable_testing()

# 添加测试
add_test(NAME tensor_create_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_tensor_create)
add_test(NAME add_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_add)
add_test(NAME sub_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_sub)
add_test(NAME mul_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_mul)
add_test(NAME div_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_div)
add_test(NAME neg_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_neg)
add_test(NAME square_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_square)
add_test(NAME exp_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_exp)
add_test(NAME pow_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_pow)
add_test(NAME sum_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_sum)
add_test(NAME sum_to_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_sum_to)
add_test(NAME broadcast_to_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_broadcast_to)
add_test(NAME transpose_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_transpose)
add_test(NAME reshape_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_reshape)
add_test(NAME matmul_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_matmul)
add_test(NAME linear_regression_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_linear_regression)

# 设置测试属性
set_tests_properties(tensor_create_test PROPERTIES TIMEOUT 60)
set_tests_properties(add_test PROPERTIES TIMEOUT 60)
set_tests_properties(sub_test PROPERTIES TIMEOUT 60)
set_tests_properties(mul_test PROPERTIES TIMEOUT 60)
set_tests_properties(div_test PROPERTIES TIMEOUT 60)
set_tests_properties(neg_test PROPERTIES TIMEOUT 60)
set_tests_properties(square_test PROPERTIES TIMEOUT 60)
set_tests_properties(exp_test PROPERTIES TIMEOUT 60)
set_tests_properties(pow_test PROPERTIES TIMEOUT 60)
set_tests_properties(sum_test PROPERTIES TIMEOUT 60)
set_tests_properties(sum_to_test PROPERTIES TIMEOUT 60)
set_tests_properties(broadcast_to_test PROPERTIES TIMEOUT 60)
set_tests_properties(transpose_test PROPERTIES TIMEOUT 60)
set_tests_properties(reshape_test PROPERTIES TIMEOUT 60)
set_tests_properties(matmul_test PROPERTIES TIMEOUT 60)
set_tests_properties(linear_regression_test PROPERTIES TIMEOUT 60)

# 添加PyTorch对比示例子目录选项
option(BUILD_PYTORCH_COMPARISON "Build PyTorch comparison examples" OFF)

if(BUILD_PYTORCH_COMPARISON)
    message(STATUS "Building PyTorch comparison examples")
    add_subdirectory(operator/pytorch_comparison)
else()
    message(STATUS "Skipping PyTorch comparison examples (BUILD_PYTORCH_COMPARISON=OFF)")
endif()

# 打印配置信息
message(STATUS "Unit tests configuration completed")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 添加自定义目标来运行所有测试
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_tensor_create test_add test_sub test_mul test_div test_neg test_square test_exp test_pow test_sum test_sum_to test_broadcast_to test_transpose test_reshape test_matmul test_linear_regression
    COMMENT "Run all unit tests"
)

# 添加自定义目标来编译和运行测试
add_custom_target(test_all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_tensor_create test_add test_sub test_mul test_div test_neg test_square test_exp test_pow test_sum test_sum_to test_broadcast_to test_transpose test_reshape test_matmul test_linear_regression
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_tensor_create test_add test_sub test_mul test_div test_neg test_square test_exp test_pow test_sum test_sum_to test_broadcast_to test_transpose test_reshape test_matmul test_linear_regression
    COMMENT "Compile and run all unit tests"
)