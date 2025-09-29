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
add_executable(test_tensor_create tesnor/test_create_tensor.cpp)
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


# 启用测试
enable_testing()

# 添加测试
add_test(NAME tensor_create_test COMMAND ${UNIT_TEST_OUTPUT_DIR}/test_tensor_create)

# 设置测试属性
set_tests_properties(tensor_create_test PROPERTIES TIMEOUT 60)

# 打印配置信息
message(STATUS "Unit tests configuration completed")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 添加自定义目标来运行所有测试
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_tensor_create
    COMMENT "Run all unit tests"
)

# 添加自定义目标来编译和运行测试
add_custom_target(test_all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_tensor_create
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_tensor_create
    COMMENT "Compile and run all unit tests"
)