cmake_minimum_required (VERSION 3.12)

# 获取最后一个目录名，并将其赋值给 PROJECT_NAME 变量
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${BIN_PREFIX}_${PROJECT_NAME})

add_compile_options(-std=c++17 -O2 -Wall -g -ggdb)

include_directories(${ORIGINDL_ROOT_DIR}/include)
# 使用主CMakeLists.txt中定义的ARRAYFIRE_PATH变量
include_directories(${ARRAYFIRE_PATH}/include)

# 编译第一个可执行程序：linear_regression
add_executable(${BIN_PREFIX}_linear_regression linear_regression.cpp)
add_dependencies(${BIN_PREFIX}_linear_regression ${MY_LIB_NAME} originlog)
target_link_libraries(${BIN_PREFIX}_linear_regression PRIVATE originlog ${MY_LIB_NAME})

# 编译第二个可执行程序：nn_linear
add_executable(${BIN_PREFIX}_nn_linear nn_linear.cpp)
add_dependencies(${BIN_PREFIX}_nn_linear ${MY_LIB_NAME} originlog)
target_link_libraries(${BIN_PREFIX}_nn_linear PRIVATE originlog ${MY_LIB_NAME})

# 仅在 ORIGIN 后端且启用 CUDA 时编译 CUDA 示例
if(ENABLE_CUDA AND MAT_BACKEND STREQUAL "ORIGIN")
    # 编译第三个可执行程序：nn_linear_cuda
    add_executable(${BIN_PREFIX}_nn_linear_cuda nn_linear_cuda.cpp)
    add_dependencies(${BIN_PREFIX}_nn_linear_cuda ${MY_LIB_NAME} originlog)
    target_link_libraries(${BIN_PREFIX}_nn_linear_cuda PRIVATE originlog ${MY_LIB_NAME} originmat_cuda)
endif()

# 设置输出目录
set_target_properties(${BIN_PREFIX}_linear_regression PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

set_target_properties(${BIN_PREFIX}_nn_linear PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

# 设置CUDA示例的输出目录
if(ENABLE_CUDA AND MAT_BACKEND STREQUAL "ORIGIN")
    set_target_properties(${BIN_PREFIX}_nn_linear_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )
endif()
