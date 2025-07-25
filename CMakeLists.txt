#minimum version required
cmake_minimum_required (VERSION 3.12)
#project name
set(MY_LIB_NAME origindl)
project (${MY_LIB_NAME})

set(DLZERO_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(
    GLOB MY_LIB_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/operator/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp 
)

# message(WARNING ${MY_LIB_SRCS})

execute_process(
    COMMAND bash "-c" "git log|head -n 1|awk '{printf $2}'"
    OUTPUT_VARIABLE GIT_COMMIT
    )
add_definitions(-DGIT_COMMIT_SHA1="${GIT_COMMIT}")

# 检查环境变量ARRAYFIRE_PATH是否存在，如果存在则使用它，否则使用默认路径
if(DEFINED ENV{ARRAYFIRE_PATH})
    set(ARRAYFIRE_PATH $ENV{ARRAYFIRE_PATH})
else()
    set(ARRAYFIRE_PATH /opt/arrayfire)
endif()

message(STATUS "Using ArrayFire path: ${ARRAYFIRE_PATH}")

include_directories(${ARRAYFIRE_PATH}/include)
include_directories(3rd/)
include_directories(include)

add_compile_options(-std=c++17 -O2 -Wall -g -ggdb)
# 禁止一些警告
# add_compile_options(-Wno-unused-variable -Wno-unused-function -Wno-reorder -Wno-format) 

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/libs/)
link_directories(${PROJECT_SOURCE_DIR}/3rd/lib)
link_directories(${ARRAYFIRE_PATH}/lib64)

# 创建spdlog静态库dllog
file(
    GLOB LOG_LIB_SRCS
        ${PROJECT_SOURCE_DIR}/3rd/spdlog/src/*.cpp
)
add_library(dllog SHARED ${LOG_LIB_SRCS})
target_include_directories(dllog PUBLIC 3rd/)
target_compile_definitions(dllog PUBLIC SPDLOG_COMPILED_LIB)

add_library(${PROJECT_NAME} SHARED ${MY_LIB_SRCS})
target_link_libraries(${PROJECT_NAME} dllog -lpthread -laf)

# 开始编译可执行程序
# 设置输出目录为 build/bin
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

# 设置传递给测试子目录的变量
set(ORIGINDL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
