cmake_minimum_required(VERSION 3.18)
project(originmat_cuda)

# originmat_cuda.so 依赖的 cpp 源文件使用 C++20 标准
# 注：CMAKE_CUDA_STANDARD 20 需要 CMake 3.25.2+
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用CUDA语言，originmat_cuda.so 依赖的 cu 源文件使用 CUDA C++20 标准
# 因为不同环境的 cuda 版本不同，支持的 C++ 标准不同，因此显式指定 C++ 标准
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA（使用现代方式）
find_package(CUDAToolkit REQUIRED)

# cuBLAS支持（由用户选项USER_USE_CUBLAS控制）
# 如果用户开启USER_USE_CUBLAS，则尝试查找cuBLAS库；若找到则启用cuBLAS支持（ENABLE_CUBLAS=ON）
# 如果用户未开启USER_USE_CUBLAS，无论环境中是否安装cuBLAS，ENABLE_CUBLAS都将为OFF
set(ENABLE_CUBLAS OFF)

# 如果用户期望使用cuBLAS，则尝试查找cuBLAS库
if(USER_USE_CUBLAS)
    # 查找cuBLAS库（尝试多个路径）
    find_library(CUBLAS_LIBRARY 
        NAMES cublas
        PATHS
            ${CUDAToolkit_LIBRARY_DIR}
            /usr/local/cuda/lib64
            /usr/lib/x86_64-linux-gnu
        NO_DEFAULT_PATH
    )

    if(CUBLAS_LIBRARY)
        message(STATUS "cuBLAS library found: ${CUBLAS_LIBRARY}")
        message(STATUS "Enabling cuBLAS support for matrix multiplication optimization")
        set(ENABLE_CUBLAS ON)
    else()
        message(WARNING "cuBLAS library not found, but USER_USE_CUBLAS is enabled. Falling back to origindl's CUDA kernels")
        set(ENABLE_CUBLAS OFF)
    endif()
else()
    message(STATUS "USER_USE_CUBLAS is disabled, using origindl's CUDA kernels")
    set(ENABLE_CUBLAS OFF)
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
# -diag-suppress=20208: 屏蔽 NVCC 警告 #20208 "long double is treated as double in device code"
#   CUDA 设备端不支持真正的 long double，会按 double 处理。该警告由 3rd/spdlog/fmt 头文件
#   在格式化中使用 long double 引发。spdlog 仅用于 CPU 日志，不参与 GPU 计算，可安全忽略。
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -rdc=true -diag-suppress=20208")

# 添加编译定义
target_compile_definitions(originmat_cuda PRIVATE WITH_CUDA)

# 如果启用cuBLAS，添加编译定义
if(ENABLE_CUBLAS)
    target_compile_definitions(originmat_cuda PRIVATE ENABLE_CUBLAS)
endif()

# 启用CUDA可重定位设备代码
set_target_properties(originmat_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(originmat_cuda CUDA::cudart CUDA::curand)

# 如果启用cuBLAS，链接cuBLAS库
if(ENABLE_CUBLAS AND CUBLAS_LIBRARY)
    target_link_libraries(originmat_cuda ${CUBLAS_LIBRARY})
    message(STATUS "Linking with cuBLAS library: ${CUBLAS_LIBRARY}")
endif()

# 设置输出目录
set_target_properties(originmat_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

message(STATUS "CUDA library will be built with architecture: ${CUDA_ARCH}")
