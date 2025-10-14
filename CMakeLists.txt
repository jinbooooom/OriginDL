#minimum version required
cmake_minimum_required (VERSION 3.12)
#project name
set(MY_LIB_NAME origindl)
project (${MY_LIB_NAME})

set(ORIGINDL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 使用GLOB自动收集源文件，避免手动维护文件列表
# 基础源文件
file(
    GLOB MY_LIB_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cpp 
)

# 根据后端添加对应的源文件
if(MAT_BACKEND STREQUAL "ARRAYFIRE")
    list(APPEND MY_LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/array_fire_mat.cpp)
    list(APPEND MY_LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/backend_mat_utils.cpp)
elseif(MAT_BACKEND STREQUAL "TORCH")
    file(GLOB TORCH_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/torch/*.cpp)
    list(APPEND MY_LIB_SRCS ${TORCH_SRCS})
elseif(MAT_BACKEND STREQUAL "ORIGIN")
    file(GLOB ORIGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/*.cpp)
    file(GLOB ORIGIN_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cpu/*.cpp)
    list(APPEND MY_LIB_SRCS ${ORIGIN_SRCS})
    list(APPEND MY_LIB_SRCS ${ORIGIN_CPU_SRCS})
endif()

# 支持多种后端选择
if(MAT_BACKEND STREQUAL "ARRAYFIRE")
    add_definitions(-DMAT_BACKEND=0)  # ARRAYFIRE = 0
elseif(MAT_BACKEND STREQUAL "TORCH")
    add_definitions(-DMAT_BACKEND=1)  # TORCH = 1
elseif(MAT_BACKEND STREQUAL "ORIGIN")
    add_definitions(-DMAT_BACKEND=2)  # ORIGIN = 2
else()
    # 默认使用TORCH后端
    add_definitions(-DMAT_BACKEND=1)  # TORCH = 1
endif()

# message(WARNING ${MY_LIB_SRCS})

# execute_process(
#     COMMAND bash "-c" "git log|head -n 1|awk '{printf $2}'"
#     OUTPUT_VARIABLE GIT_COMMIT
#     )
# add_definitions(-DGIT_COMMIT_SHA1="${GIT_COMMIT}")

# 检查环境变量ARRAYFIRE_PATH是否存在，如果存在则使用它，否则使用默认路径
if(DEFINED ENV{ARRAYFIRE_PATH})
    set(ARRAYFIRE_PATH $ENV{ARRAYFIRE_PATH})
else()
    set(ARRAYFIRE_PATH /opt/arrayfire)
endif()

# 检查环境变量TORCH_PATH是否存在，如果存在则使用它，否则使用默认路径
if(DEFINED ENV{TORCH_PATH})
    set(TORCH_PATH $ENV{TORCH_PATH})
else()
    set(TORCH_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd/libtorch)
endif()

message(STATUS "Using ArrayFire path: ${ARRAYFIRE_PATH}")
message(STATUS "Using LibTorch path: ${TORCH_PATH}")

include_directories(${ARRAYFIRE_PATH}/include)
include_directories(3rd/)
include_directories(include)

# 根据后端选择包含LibTorch
if(MAT_BACKEND STREQUAL "TORCH")
    # 设置LibTorch的cmake路径
    set(Torch_DIR "${TORCH_PATH}/share/cmake/Torch")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    message(STATUS "LibTorch version: ${Torch_VERSION}")
    message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
endif()

add_compile_options(-std=c++17 -O2 -Wall -g -ggdb)
# 禁止一些警告
# add_compile_options(-Wno-unused-variable -Wno-unused-function -Wno-reorder -Wno-format) 

# 根据后端动态设置库输出目录
if(MAT_BACKEND STREQUAL "ORIGIN")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib/)
elseif(MAT_BACKEND STREQUAL "TORCH")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/torch_build/lib/)
elseif(MAT_BACKEND STREQUAL "ARRAYFIRE")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/arrayfire_build/lib/)
else()
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib/)
endif()
link_directories(${PROJECT_SOURCE_DIR}/3rd/lib)
link_directories(${ARRAYFIRE_PATH}/lib64)

# 根据后端添加相应的库目录
if(MAT_BACKEND STREQUAL "TORCH")
    link_directories(${TORCH_PATH}/lib)
endif()

# 编译GoogleTest库，供所有单元测试使用
message(STATUS "Building GoogleTest libraries")
add_subdirectory(${PROJECT_SOURCE_DIR}/3rd/googletest ${PROJECT_BINARY_DIR}/googletest_build)

# 设置GoogleTest相关变量，供子项目使用
# 直接使用GoogleTest编译后的库文件路径
set(GTEST_LIBRARIES 
    ${PROJECT_BINARY_DIR}/lib/libgtest.a
    ${PROJECT_BINARY_DIR}/lib/libgtest_main.a
    pthread
)
set(GTEST_INCLUDE_DIRS 
    ${PROJECT_SOURCE_DIR}/3rd/googletest/googletest/include
    ${PROJECT_SOURCE_DIR}/3rd/googletest/googlemock/include
)

message(STATUS "GoogleTest libraries will be built and available for all tests")

# 创建spdlog静态库dllog
file(
    GLOB LOG_LIB_SRCS
        ${PROJECT_SOURCE_DIR}/3rd/spdlog/src/*.cpp
)
add_library(dllog SHARED ${LOG_LIB_SRCS})
target_include_directories(dllog PUBLIC 3rd/)
target_compile_definitions(dllog PUBLIC SPDLOG_COMPILED_LIB)

add_library(${PROJECT_NAME} SHARED ${MY_LIB_SRCS})

# 根据后端选择相应的链接库
if(MAT_BACKEND STREQUAL "TORCH")
    target_link_libraries(${PROJECT_NAME} dllog -lpthread "${TORCH_LIBRARIES}")
    message(STATUS "Linking with LibTorch backend")
elseif(MAT_BACKEND STREQUAL "ARRAYFIRE")
    target_link_libraries(${PROJECT_NAME} dllog -lpthread -laf)
    message(STATUS "Linking with ArrayFire backend")
elseif(MAT_BACKEND STREQUAL "ORIGIN")
    target_link_libraries(${PROJECT_NAME} dllog -lpthread)
    message(STATUS "Linking with OriginMat backend")
else()
    target_link_libraries(${PROJECT_NAME} dllog -lpthread)
    message(STATUS "Linking with default backend")
endif()

# 开始编译可执行程序
# 根据后端动态设置输出目录
if(MAT_BACKEND STREQUAL "ORIGIN")
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
elseif(MAT_BACKEND STREQUAL "TORCH")
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/torch_build/bin)
elseif(MAT_BACKEND STREQUAL "ARRAYFIRE")
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/arrayfire_build/bin)
else()
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
endif()

# 设置传递给测试子目录的变量
set(ORIGINDL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
