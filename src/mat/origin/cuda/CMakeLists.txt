cmake_minimum_required(VERSION 3.18)
project(originmat_cuda)

# originmat_cuda.so 依赖的 cpp 源文件使用 C++11 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用CUDA语言，originmat_cuda.so 依赖的 cu 源文件使用 CUDA C++11 标准
# 因为不同环境的 cuda 版本不同，支持的 C++ 标准不同，因此显式指定 C++ 标准
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA（使用现代方式）
find_package(CUDAToolkit REQUIRED)

# 查找 cuDNN
# 优先查找 cuDNN（支持多种安装路径）
find_path(CUDNN_INCLUDE_DIR cudnn.h
    PATHS
    $ENV{CUDNN_ROOT}/include
    $ENV{CUDNN_HOME}/include
    /usr/local/cuda/include
    /usr/include
    /usr/local/include
    $ENV{CUDA_PATH}/include
    $ENV{CUDA_HOME}/include
    NO_DEFAULT_PATH
)

# 如果第一次没找到，再尝试默认路径
if(NOT CUDNN_INCLUDE_DIR)
    find_path(CUDNN_INCLUDE_DIR cudnn.h)
endif()

find_library(CUDNN_LIBRARY
    NAMES cudnn libcudnn.so.8 libcudnn.so.7 libcudnn.so
    PATHS
    $ENV{CUDNN_ROOT}/lib
    $ENV{CUDNN_ROOT}/lib64
    $ENV{CUDNN_HOME}/lib
    $ENV{CUDNN_HOME}/lib64
    /usr/local/cuda/lib64
    /usr/local/cuda/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/lib
    $ENV{CUDA_PATH}/lib64
    $ENV{CUDA_HOME}/lib64
    NO_DEFAULT_PATH
)

# 如果第一次没找到，再尝试默认路径
if(NOT CUDNN_LIBRARY)
    find_library(CUDNN_LIBRARY NAMES cudnn libcudnn.so.8 libcudnn.so.7 libcudnn.so)
endif()

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    message(STATUS "Found cuDNN:")
    message(STATUS "  Include: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "  Library: ${CUDNN_LIBRARY}")
    set(HAVE_CUDNN ON)
else()
    message(WARNING "cuDNN not found, convolution will use im2col + cuBLAS GEMM")
    message(WARNING "  To enable cuDNN, set CUDNN_ROOT environment variable to cuDNN installation directory")
    message(WARNING "  Example: export CUDNN_ROOT=/usr/local/cuda")
    set(HAVE_CUDNN OFF)
endif()

# 自动检测CUDA架构
function(detect_cuda_architecture)
    # 尝试使用nvidia-smi检测GPU架构
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        RESULT_VARIABLE NVIDIA_SMI_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
        # 获取第一行的计算能力（处理多GPU情况）
        string(REGEX REPLACE "\n.*" "" FIRST_GPU_CAP "${GPU_COMPUTE_CAP}")
        # 解析计算能力（例如：8.0 -> 80）
        string(REGEX REPLACE "\\." "" COMPUTE_CAP_NUM "${FIRST_GPU_CAP}")
        set(CUDA_ARCH "sm_${COMPUTE_CAP_NUM}" PARENT_SCOPE)
        message(STATUS "Auto-detected CUDA architecture: sm_${COMPUTE_CAP_NUM} (from nvidia-smi)")
    else()
        # 如果检测失败，使用默认架构
        set(CUDA_ARCH "sm_75" PARENT_SCOPE)
        message(WARNING "Failed to auto-detect CUDA architecture, using default: sm_75")
    endif()
endfunction()

# 检测CUDA架构
detect_cuda_architecture()

# 收集CUDA/CPP源文件（支持子目录）
file(GLOB_RECURSE CUDA_SRCS CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)
file(GLOB_RECURSE CUDA_CPP_SRCS CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# 创建CUDA库（包含本目录下所有 .cu 与 .cpp 文件）
add_library(originmat_cuda SHARED ${CUDA_SRCS} ${CUDA_CPP_SRCS})

# 设置包含目录
target_include_directories(originmat_cuda PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rd
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 设置CUDA编译选项
# 从CUDA_ARCH中提取数字部分（例如：sm_80 -> 80）
string(REGEX REPLACE "sm_" "" CUDA_ARCH_NUM "${CUDA_ARCH}")
set_target_properties(originmat_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_NUM}"
)

# 设置CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -rdc=true")

# 添加编译定义
target_compile_definitions(originmat_cuda PRIVATE WITH_CUDA)

# 启用CUDA可重定位设备代码
set_target_properties(originmat_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库（包括cuBLAS）
target_link_libraries(originmat_cuda CUDA::cudart CUDA::curand CUDA::cublas)

# 如果找到 cuDNN，链接它
if(HAVE_CUDNN)
    target_include_directories(originmat_cuda PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(originmat_cuda ${CUDNN_LIBRARY})
    target_compile_definitions(originmat_cuda PRIVATE HAVE_CUDNN)
    message(STATUS "cuDNN enabled for convolution operations")
endif()

# 设置输出目录
set_target_properties(originmat_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../../build/lib
)

message(STATUS "CUDA library will be built with architecture: ${CUDA_ARCH}")
