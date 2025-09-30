# GoogleTest Demo CMakeLists.txt
# 这个文件演示如何配置GoogleTest进行单元测试

cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(3rd_googletest_project)

# 继承主项目的编译选项
message(STATUS "3rd_googletest inheriting compilation options from main project")

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 包含GoogleTest
# 使用主项目编译的GoogleTest库（只能从主项目调用）
# 继承主项目设置的GoogleTest变量
message(STATUS "Using GoogleTest libraries compiled by main project")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 创建测试可执行文件（Calculator类实现已直接包含在测试文件中）
add_executable(3rd_googletest
    googletest.cpp
)

# 设置输出目录
set_target_properties(3rd_googletest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

# 确保依赖GoogleTest库的编译
add_dependencies(3rd_googletest gtest gtest_main)

# 链接库
target_link_libraries(3rd_googletest
    ${GTEST_LIBRARIES}
    pthread
)

# 设置包含目录
target_include_directories(3rd_googletest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GTEST_INCLUDE_DIRS}
)

# 启用测试
enable_testing()

# 打印配置信息
message(STATUS "GoogleTest Demo configuration completed")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 添加自定义目标来运行测试
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS 3rd_googletest
    COMMENT "Run all tests"
)

# 添加自定义目标来编译和运行测试
add_custom_target(demo
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target 3rd_googletest
    COMMAND ${CMAKE_BINARY_DIR}/3rd_googletest
    DEPENDS 3rd_googletest
    COMMENT "Compile and run GoogleTest Demo"
)
