cmake_minimum_required (VERSION 3.12)

# 获取最后一个目录名，并将其赋值给 PROJECT_NAME 变量
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${BIN_PREFIX}_${PROJECT_NAME})

include_directories(${ORIGINDL_ROOT_DIR}/include)
# 使用主CMakeLists.txt中定义的ARRAYFIRE_PATH变量
include_directories(${ARRAYFIRE_PATH}/include)

# 编译第一个可执行程序：tensor
add_executable(${BIN_PREFIX}_tensor tensor.cpp)
add_dependencies(${BIN_PREFIX}_tensor ${MY_LIB_NAME})
target_link_libraries(${BIN_PREFIX}_tensor PRIVATE ${MY_LIB_NAME})

# 编译第二个可执行程序：tensor_print
add_executable(${BIN_PREFIX}_tensor_print tensor_print.cpp)
add_dependencies(${BIN_PREFIX}_tensor_print ${MY_LIB_NAME})
target_link_libraries(${BIN_PREFIX}_tensor_print PRIVATE ${MY_LIB_NAME})

# 编译第三个可执行程序：tensor_cuda (仅在启用CUDA时)
if(ENABLE_CUDA)
    add_executable(${BIN_PREFIX}_tensor_cuda tensor_cuda.cpp)
    add_dependencies(${BIN_PREFIX}_tensor_cuda ${MY_LIB_NAME})
    target_link_libraries(${BIN_PREFIX}_tensor_cuda PRIVATE ${MY_LIB_NAME})
    
    # 编译第四个可执行程序：tensor_cuda_extended (仅在启用CUDA时)
    add_executable(${BIN_PREFIX}_tensor_cuda_extended tensor_cuda_extended.cpp)
    add_dependencies(${BIN_PREFIX}_tensor_cuda_extended ${MY_LIB_NAME})
    target_link_libraries(${BIN_PREFIX}_tensor_cuda_extended PRIVATE ${MY_LIB_NAME})
    
    # 编译第五个可执行程序：tensor_cuda_debug (仅在启用CUDA时)
    add_executable(${BIN_PREFIX}_tensor_cuda_debug tensor_cuda_debug.cpp)
    add_dependencies(${BIN_PREFIX}_tensor_cuda_debug ${MY_LIB_NAME})
    target_link_libraries(${BIN_PREFIX}_tensor_cuda_debug PRIVATE ${MY_LIB_NAME})
    
    # 编译第六个可执行程序：tensor_cuda_simple (仅在启用CUDA时)
    add_executable(${BIN_PREFIX}_tensor_cuda_simple tensor_cuda_simple.cpp)
    add_dependencies(${BIN_PREFIX}_tensor_cuda_simple ${MY_LIB_NAME})
    target_link_libraries(${BIN_PREFIX}_tensor_cuda_simple PRIVATE ${MY_LIB_NAME})
    
    # 编译第七个可执行程序：tensor_cuda_minimal (仅在启用CUDA时)
    add_executable(${BIN_PREFIX}_tensor_cuda_minimal tensor_cuda_minimal.cpp)
    add_dependencies(${BIN_PREFIX}_tensor_cuda_minimal ${MY_LIB_NAME})
    target_link_libraries(${BIN_PREFIX}_tensor_cuda_minimal PRIVATE ${MY_LIB_NAME})
    
    # 如果使用ORIGIN后端，还需要链接CUDA库
    if(MAT_BACKEND STREQUAL "ORIGIN")
        target_link_libraries(${BIN_PREFIX}_tensor_cuda PRIVATE originmat_cuda)
        target_link_libraries(${BIN_PREFIX}_tensor_cuda_extended PRIVATE originmat_cuda)
        target_link_libraries(${BIN_PREFIX}_tensor_cuda_debug PRIVATE originmat_cuda)
    endif()
endif()

# 设置输出目录
set_target_properties(${BIN_PREFIX}_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

set_target_properties(${BIN_PREFIX}_tensor_print PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

# 设置CUDA示例的输出目录
if(ENABLE_CUDA)
    set_target_properties(${BIN_PREFIX}_tensor_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )
    set_target_properties(${BIN_PREFIX}_tensor_cuda_extended PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )
    set_target_properties(${BIN_PREFIX}_tensor_cuda_debug PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )
endif()
