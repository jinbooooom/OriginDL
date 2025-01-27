#minimum version required
cmake_minimum_required (VERSION 3.12)
#project name
set(MY_LIB_NAME origindl)
project (${MY_LIB_NAME})

set(DLZERO_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(
    GLOB MY_LIB_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
)

# message(WARNING ${MY_LIB_SRCS})

execute_process(
    COMMAND bash "-c" "git log|head -n 1|awk '{printf $2}'"
    OUTPUT_VARIABLE GIT_COMMIT
    )
add_definitions(-DGIT_COMMIT_SHA1="${GIT_COMMIT}")

include_directories(3rd/numCpp)
include_directories(3rd/)
include_directories(include)

add_compile_options(-std=c++17 -O2 -Wall -g -ggdb)
# 禁止一些警告
# add_compile_options(-Wno-unused-variable -Wno-unused-function -Wno-reorder -Wno-format) 

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/libs/)
link_directories(${PROJECT_SOURCE_DIR}/3rd/lib)

add_library(${PROJECT_NAME} SHARED ${MY_LIB_SRCS})
target_link_libraries(${PROJECT_NAME} -lpthread)

# 开始编译可执行程序
# 设置输出目录为 build/bin
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)