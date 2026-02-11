cmake_minimum_required (VERSION 3.12)

# 获取最后一个目录名，并将其赋值给 PROJECT_NAME 变量
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${BIN_PREFIX}_${PROJECT_NAME})

add_compile_options(-std=c++20 -O2 -Wall -g -ggdb)

include_directories(${ORIGINDL_ROOT_DIR}/include)
# 使用主CMakeLists.txt中定义的ARRAYFIRE_PATH变量
include_directories(${ARRAYFIRE_PATH}/include)

add_executable(${BIN_PREFIX}_linear_regression linear_regression.cpp)
add_dependencies(${BIN_PREFIX}_linear_regression ${MY_LIB_NAME} originlog)
target_link_libraries(${BIN_PREFIX}_linear_regression PRIVATE originlog ${MY_LIB_NAME})

add_executable(${BIN_PREFIX}_nn_linear nn_linear.cpp)
add_dependencies(${BIN_PREFIX}_nn_linear ${MY_LIB_NAME} originlog)
target_link_libraries(${BIN_PREFIX}_nn_linear PRIVATE originlog ${MY_LIB_NAME})

# 仅在 ORIGIN 后端且启用 CUDA 时链接 CUDA 库
if(ENABLE_CUDA AND MAT_BACKEND STREQUAL "ORIGIN")
    target_link_libraries(${BIN_PREFIX}_nn_linear PRIVATE originmat_cuda)
endif()

# 设置输出目录
set_target_properties(${BIN_PREFIX}_linear_regression PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

set_target_properties(${BIN_PREFIX}_nn_linear PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)
