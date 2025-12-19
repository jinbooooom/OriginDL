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
add_dependencies(test_tensor_create gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_tensor_create 
    ${GTEST_LIBRARIES}
    origindl
    test_utils
    gtest_utils
    pthread
)
target_include_directories(test_tensor_create PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests/common
    ${GTEST_INCLUDE_DIRS}
)
set_target_properties(test_tensor_create PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_no_grad core/test_no_grad.cpp)
add_dependencies(test_no_grad gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_no_grad ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_no_grad PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_no_grad PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 添加多类型张量测试
add_executable(test_multi_type_tensor tensor/test_multi_type_tensor.cpp)
add_dependencies(test_multi_type_tensor gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_multi_type_tensor 
    ${GTEST_LIBRARIES}
    origindl
    test_utils
    gtest_utils
    pthread
)
target_include_directories(test_multi_type_tensor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests/common
    ${GTEST_INCLUDE_DIRS}
)
set_target_properties(test_multi_type_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 添加TensorOptions测试
add_executable(test_tensor_options tensor/test_tensor_options.cpp)
add_dependencies(test_tensor_options gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_tensor_options 
    ${GTEST_LIBRARIES}
    origindl
    test_utils
    gtest_utils
    pthread
)
target_include_directories(test_tensor_options PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests/common
    ${GTEST_INCLUDE_DIRS}
)
set_target_properties(test_tensor_options PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 添加张量属性测试
add_executable(test_tensor_attributes tensor/test_tensor_attributes.cpp)
add_dependencies(test_tensor_attributes gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_tensor_attributes ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_tensor_attributes PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_tensor_attributes PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 添加张量操作方法测试
add_executable(test_tensor_operations tensor/test_tensor_operations.cpp)
add_dependencies(test_tensor_operations gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_tensor_operations ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_tensor_operations PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_tensor_operations PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 添加算子测试
# 基础算术算子测试
add_executable(test_add operator/test_add.cpp)
add_dependencies(test_add gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_add ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_add PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_add PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_sub operator/test_sub.cpp)
add_dependencies(test_sub gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_sub ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_sub PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_sub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_mul operator/test_mul.cpp)
add_dependencies(test_mul gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_mul ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_mul PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_mul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_div operator/test_div.cpp)
add_dependencies(test_div gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_div ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_div PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_div PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 一元算子测试
add_executable(test_neg operator/test_neg.cpp)
add_dependencies(test_neg gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_neg ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_neg PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_neg PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_square operator/test_square.cpp)
add_dependencies(test_square gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_square ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_square PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_square PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_exp operator/test_exp.cpp)
add_dependencies(test_exp gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_exp ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_exp PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_exp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_log operator/test_log.cpp)
add_dependencies(test_log gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_log ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_log PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_log PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_softmax operator/test_softmax.cpp)
add_dependencies(test_softmax gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_softmax ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_softmax PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_softmax PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_relu operator/test_relu.cpp)
add_dependencies(test_relu gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_relu ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_relu PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_relu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_softmax_cross_entropy operator/test_softmax_cross_entropy.cpp)
add_dependencies(test_softmax_cross_entropy gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_softmax_cross_entropy ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_softmax_cross_entropy PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_softmax_cross_entropy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_accuracy utils/test_accuracy.cpp)
add_dependencies(test_accuracy gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_accuracy ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_accuracy PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_accuracy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_adam optim/test_adam.cpp)
add_dependencies(test_adam gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_adam ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_adam PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_adam PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_weight_decay optim/test_weight_decay.cpp)
add_dependencies(test_weight_decay gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_weight_decay ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_weight_decay PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_weight_decay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_mlp nn/test_mlp.cpp)
add_dependencies(test_mlp gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_mlp ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_mlp PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_mlp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_pow operator/test_pow.cpp)
add_dependencies(test_pow gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_pow ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_pow PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_pow PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 添加类型提升测试
add_executable(test_type_promotion operator/test_type_promotion.cpp)
add_dependencies(test_type_promotion gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_type_promotion ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_type_promotion PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_type_promotion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 聚合算子测试
add_executable(test_sum operator/test_sum.cpp)
add_dependencies(test_sum gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_sum ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_sum PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_sum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_sum_to operator/test_sum_to.cpp)
add_dependencies(test_sum_to gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_sum_to ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_sum_to PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_sum_to PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_broadcast_to operator/test_broadcast_to.cpp)
add_dependencies(test_broadcast_to gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_broadcast_to ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_broadcast_to PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_broadcast_to PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 形状操作算子测试
add_executable(test_transpose operator/test_transpose.cpp)
add_dependencies(test_transpose gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_transpose ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_transpose PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_transpose PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

add_executable(test_reshape operator/test_reshape.cpp)
add_dependencies(test_reshape gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_reshape ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_reshape PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_reshape PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 矩阵运算算子测试
add_executable(test_matmul operator/test_matmul.cpp)
add_dependencies(test_matmul gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_matmul ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_matmul PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_matmul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 自动微分测试
add_executable(test_linear_regression autograd/test_linear_regression.cpp)
add_dependencies(test_linear_regression gtest gtest_main origindl test_utils gtest_utils)
target_link_libraries(test_linear_regression ${GTEST_LIBRARIES} origindl test_utils gtest_utils pthread)
target_include_directories(test_linear_regression PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/common ${GTEST_INCLUDE_DIRS})
set_target_properties(test_linear_regression PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)


# 启用测试
enable_testing()

# 添加测试
add_test(NAME tensor_create_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_tensor_create)
add_test(NAME no_grad_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_no_grad)
add_test(NAME multi_type_tensor_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_multi_type_tensor)
add_test(NAME tensor_options_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_tensor_options)
add_test(NAME tensor_attributes_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_tensor_attributes)
add_test(NAME tensor_operations_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_tensor_operations)
add_test(NAME add_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_add)
add_test(NAME sub_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_sub)
add_test(NAME mul_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_mul)
add_test(NAME div_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_div)
add_test(NAME neg_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_neg)
add_test(NAME square_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_square)
add_test(NAME exp_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_exp)
add_test(NAME log_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_log)
add_test(NAME softmax_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_softmax)
add_test(NAME relu_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_relu)
add_test(NAME softmax_cross_entropy_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_softmax_cross_entropy)
add_test(NAME accuracy_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_accuracy)
add_test(NAME adam_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_adam)
add_test(NAME weight_decay_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_weight_decay)
add_test(NAME mlp_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_mlp)
add_test(NAME pow_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_pow)
add_test(NAME type_promotion_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_type_promotion)
add_test(NAME sum_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_sum)
add_test(NAME sum_to_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_sum_to)
add_test(NAME broadcast_to_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_broadcast_to)
add_test(NAME transpose_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_transpose)
add_test(NAME reshape_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_reshape)
add_test(NAME matmul_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_matmul)
add_test(NAME linear_regression_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_linear_regression)

# 设置测试属性
set_tests_properties(tensor_create_test PROPERTIES TIMEOUT 60)
set_tests_properties(multi_type_tensor_test PROPERTIES TIMEOUT 60)
set_tests_properties(tensor_options_test PROPERTIES TIMEOUT 60)
set_tests_properties(tensor_attributes_test PROPERTIES TIMEOUT 60)
set_tests_properties(tensor_operations_test PROPERTIES TIMEOUT 60)
set_tests_properties(add_test PROPERTIES TIMEOUT 60)
set_tests_properties(sub_test PROPERTIES TIMEOUT 60)
set_tests_properties(mul_test PROPERTIES TIMEOUT 60)
set_tests_properties(div_test PROPERTIES TIMEOUT 60)
set_tests_properties(neg_test PROPERTIES TIMEOUT 60)
set_tests_properties(square_test PROPERTIES TIMEOUT 60)
set_tests_properties(exp_test PROPERTIES TIMEOUT 60)
set_tests_properties(log_test PROPERTIES TIMEOUT 60)
set_tests_properties(pow_test PROPERTIES TIMEOUT 60)
set_tests_properties(type_promotion_test PROPERTIES TIMEOUT 60)
set_tests_properties(sum_test PROPERTIES TIMEOUT 60)
set_tests_properties(sum_to_test PROPERTIES TIMEOUT 60)
set_tests_properties(broadcast_to_test PROPERTIES TIMEOUT 60)
set_tests_properties(transpose_test PROPERTIES TIMEOUT 60)
set_tests_properties(reshape_test PROPERTIES TIMEOUT 60)
set_tests_properties(matmul_test PROPERTIES TIMEOUT 60)
set_tests_properties(linear_regression_test PROPERTIES TIMEOUT 60)

# 打印配置信息
message(STATUS "Unit tests configuration completed")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 添加自定义目标来运行所有测试
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_tensor_create test_multi_type_tensor test_tensor_options test_tensor_attributes test_tensor_operations test_add test_sub test_mul test_div test_neg test_square test_exp test_log test_softmax test_relu test_softmax_cross_entropy test_accuracy test_adam test_pow test_type_promotion test_sum test_sum_to test_broadcast_to test_transpose test_reshape test_matmul test_linear_regression
    COMMENT "Run all unit tests"
)

# 添加自定义目标来编译和运行测试
add_custom_target(test_all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_tensor_create test_multi_type_tensor test_tensor_options test_tensor_attributes test_tensor_operations test_add test_sub test_mul test_div test_neg test_square test_exp test_pow test_type_promotion test_sum test_sum_to test_broadcast_to test_transpose test_reshape test_matmul test_linear_regression
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_tensor_create test_multi_type_tensor test_tensor_options test_tensor_attributes test_tensor_operations test_add test_sub test_mul test_div test_neg test_square test_exp test_log test_softmax test_relu test_softmax_cross_entropy test_accuracy test_adam test_pow test_type_promotion test_sum test_sum_to test_broadcast_to test_transpose test_reshape test_matmul test_linear_regression
    COMMENT "Compile and run all unit tests"
)