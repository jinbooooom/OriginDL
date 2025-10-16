cmake_minimum_required(VERSION 3.18)
project(originmat_cuda)

# originmat_cuda.so 依赖的 cpp 源文件使用 C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用CUDA语言，originmat_cuda.so 依赖的 cu 源文件使用 CUDA C++11 标准
# 因为不同环境的 cuda 版本不同，支持的 C++ 标准不同，因此显式指定 C++ 标准
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA（使用现代方式）
find_package(CUDAToolkit REQUIRED)

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

# 收集CUDA源文件
file(GLOB CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB CUDA_CPP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 确保包含所有新实现的算子文件
set(CUDA_OPERATOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/add.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/subtract.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/multiply.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/divide.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/exp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/log.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/sqrt.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/square.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/negate.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/add_scalar.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/multiply_scalar.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu
)

# 添加其他CUDA源文件
set(CUDA_OTHER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_broadcast.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/device_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/stream_manager.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/factory.cu
)

# 合并所有源文件
list(APPEND CUDA_SRCS ${CUDA_OPERATOR_SRCS})
list(APPEND CUDA_SRCS ${CUDA_OTHER_SRCS})

# 创建CUDA库
add_library(originmat_cuda SHARED ${CUDA_SRCS})

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

# 链接CUDA库
target_link_libraries(originmat_cuda CUDA::cudart CUDA::curand)

# 设置输出目录
set_target_properties(originmat_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../../build/lib
)

message(STATUS "CUDA library will be built with architecture: ${CUDA_ARCH}")
