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

# 设置输出目录
set_target_properties(${BIN_PREFIX}_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

set_target_properties(${BIN_PREFIX}_tensor_print PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)
