# 公共测试工具库 CMakeLists.txt
# 提供通用的测试工具，不依赖GoogleTest（test_utils）和依赖GoogleTest的工具（gtest_utils）

cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(test_common)

# 继承主项目的编译选项
message(STATUS "test_common inheriting compilation options from main project")

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 设置包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# 通用测试工具库（不依赖GoogleTest）
# ============================================================================

# 创建通用测试工具库
add_library(test_utils STATIC test_utils.cpp)

target_link_libraries(test_utils origindl)

target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 如果启用CUDA，链接CUDA库
if(ENABLE_CUDA)
    if(TARGET originmat_cuda)
        target_link_libraries(test_utils originmat_cuda)
    endif()
endif()

# ============================================================================
# GoogleTest专用工具库（依赖GoogleTest）
# ============================================================================

# 使用主项目编译的GoogleTest库（继承主项目设置的变量）
message(STATUS "Using GoogleTest libraries compiled by main project")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GoogleTest include directories: ${GTEST_INCLUDE_DIRS}")

# 检查GoogleTest是否可用
if(GTEST_LIBRARIES AND GTEST_INCLUDE_DIRS)
    # 创建GoogleTest专用工具库
    add_library(gtest_utils STATIC gtest_utils.cpp)

    # 确保依赖GoogleTest库的编译
    add_dependencies(gtest_utils gtest gtest_main)

    target_link_libraries(gtest_utils test_utils ${GTEST_LIBRARIES} pthread)

    target_include_directories(gtest_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${GTEST_INCLUDE_DIRS})

    message(STATUS "GoogleTest utilities library created")
else()
    message(WARNING "GoogleTest not found, gtest_utils library will not be created")
endif()

# 打印配置信息
message(STATUS "Test common utilities configuration completed")
message(STATUS "Test utils library: test_utils")
if(TARGET gtest_utils)
    message(STATUS "GTest utils library: gtest_utils")
endif()

