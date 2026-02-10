cmake_minimum_required(VERSION 3.18)
project(originmat_cuda)

# originmat_cuda.so 依赖的 cpp 源文件使用 C++20 标准
# 注：CMAKE_CUDA_STANDARD 20 需要 CMake 3.25.2+
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 CUDA 语言（.cu 用 nvcc 编译）。cu 源文件实际使用 C++20，由下方 target_compile_options 指定。
#
# 为何这里设 CMAKE_CUDA_STANDARD=17 而不是 20？
# 若让 CMake 使用 "CUDA20" 方言，部分环境会报错："Target requires the language dialect CUDA20,
# but CMake does not know the compile flags to enable it"。因此这里故意设为 17，让 CMake 只生成
# CUDA17 相关逻辑，不请求 CUDA20；真正的 C++20 由后面的 target_compile_options(-std=c++20) 覆盖。
#
# CMAKE_CUDA_STANDARD_REQUIRED OFF：表示 17 非强制，允许被编译选项覆盖。
# enable_language(CUDA)：启用 CUDA，使 CMake 识别 .cu 并调用 nvcc。
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)
enable_language(CUDA)

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

# 设置 CUDA 编译选项（仅对 .cu 生效，由 COMPILE_LANGUAGE:CUDA 限定）
#
# 此处才是 CUDA 使用 C++20 编译的来源：-std=c++20 传给 nvcc，覆盖前面 CMAKE_CUDA_STANDARD=17。
# 其它选项：
#   --expt-relaxed-constexpr：放宽设备端 constexpr 限制
#   -rdc=true：可重定位设备代码，配合 CUDA_SEPARABLE_COMPILATION
#   -diag-suppress=20208：屏蔽 "long double is treated as double in device code"；设备端无真正
#     long double，该警告由 3rd/spdlog/fmt 引发，spdlog 仅用于 CPU 日志，可安全忽略
target_compile_options(originmat_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20 --expt-relaxed-constexpr -rdc=true -diag-suppress=20208>
)

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
