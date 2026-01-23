#minimum version required
cmake_minimum_required (VERSION 3.18)
#project name
set(MY_LIB_NAME origindl)
project (${MY_LIB_NAME})

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA支持选项（不设置默认架构，由CUDA子项目自动检测）
option(ENABLE_CUDA "Enable CUDA support for OriginMat" OFF)
# cuBLAS支持选项（仅在启用CUDA时有效）
# 如果用户开启此选项，CUDA子项目将尝试查找cuBLAS库；若找到则启用cuBLAS支持（ENABLE_CUBLAS=ON）
# 如果用户未开启此选项，无论环境中是否安装cuBLAS，ENABLE_CUBLAS都将为OFF
# 这样可以测试自定义的 CUDA 内核的性能，而不是使用 cuBLAS 库
option(USER_USE_CUBLAS "User option to enable cuBLAS" OFF)

set(ORIGINDL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 使用GLOB自动收集源文件，避免手动维护文件列表
# 基础源文件
file(
    GLOB MY_LIB_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/**/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/**/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pnnx/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pnnx/**/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.cpp
)

# 根据后端添加对应的源文件
if(MAT_BACKEND STREQUAL "TORCH")
    file(GLOB TORCH_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/torch/*.cpp)
    list(APPEND MY_LIB_SRCS ${TORCH_SRCS})
elseif(MAT_BACKEND STREQUAL "ORIGIN")
    file(GLOB ORIGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/*.cpp)
    # 使用 GLOB_RECURSE 收集 CPU 算子（支持子目录）
    file(GLOB_RECURSE ORIGIN_CPU_SRCS 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cpu/*.cpp
    )
    # 收集 memory 目录下的源文件
    file(GLOB ORIGIN_MEMORY_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/memory/*.cpp)
    list(APPEND MY_LIB_SRCS ${ORIGIN_SRCS})
    list(APPEND MY_LIB_SRCS ${ORIGIN_CPU_SRCS})
    list(APPEND MY_LIB_SRCS ${ORIGIN_MEMORY_SRCS})
    
    # 如果启用CUDA，添加CUDA子项目
    if(ENABLE_CUDA)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cuda)
    endif()
endif()

# 支持多种后端选择
if(MAT_BACKEND STREQUAL "TORCH")
    add_definitions(-DMAT_BACKEND=1)  # TORCH = 1
elseif(MAT_BACKEND STREQUAL "ORIGIN")
    add_definitions(-DMAT_BACKEND=0)  # ORIGIN = 0
    if(ENABLE_CUDA)
        add_definitions(-DWITH_CUDA)
    endif()
else()
    # 默认使用ORIGIN后端
    add_definitions(-DMAT_BACKEND=0)  # ORIGIN = 0
endif()

# CUDA支持现在由独立的子项目处理

# message(WARNING ${MY_LIB_SRCS})

# execute_process(
#     COMMAND bash "-c" "git log|head -n 1|awk '{printf $2}'"
#     OUTPUT_VARIABLE GIT_COMMIT
#     )
# add_definitions(-DGIT_COMMIT_SHA1="${GIT_COMMIT}")

# 检查环境变量TORCH_PATH是否存在，如果存在则使用它，否则使用默认路径
if(DEFINED ENV{TORCH_PATH})
    set(TORCH_PATH $ENV{TORCH_PATH})
else()
    set(TORCH_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd/libtorch)
endif()

message(STATUS "Using LibTorch path: ${TORCH_PATH}")
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

# 根据后端设置输出目录
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/)
link_directories(${PROJECT_SOURCE_DIR}/3rd/lib)

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

# 创建日志库 originlog
file(
    GLOB LOG_LIB_SRCS
        ${PROJECT_SOURCE_DIR}/3rd/spdlog/src/*.cpp
)
add_library(originlog SHARED ${LOG_LIB_SRCS})
target_include_directories(originlog PUBLIC 3rd/)
target_compile_definitions(originlog PUBLIC SPDLOG_COMPILED_LIB)

add_library(${PROJECT_NAME} SHARED ${MY_LIB_SRCS})

# 根据后端选择相应的链接库
if(MAT_BACKEND STREQUAL "TORCH")
    target_link_libraries(${PROJECT_NAME} originlog -lpthread "${TORCH_LIBRARIES}")
    message(STATUS "Linking with LibTorch backend")
elseif(MAT_BACKEND STREQUAL "ORIGIN")
    target_link_libraries(${PROJECT_NAME} originlog -lpthread)
    message(STATUS "Linking with OriginMat backend")
    
    # 如果启用CUDA，链接CUDA库
    if(ENABLE_CUDA)
        target_link_libraries(${PROJECT_NAME} originmat_cuda)
        message(STATUS "Linking with CUDA library")
    endif()
else()
    target_link_libraries(${PROJECT_NAME} originlog -lpthread)
    message(STATUS "Linking with default backend")
endif()

# 根据后端设置可执行文件输出目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# 设置传递给测试子目录的变量
set(ORIGINDL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
