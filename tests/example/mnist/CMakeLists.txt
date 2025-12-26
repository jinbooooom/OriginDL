# MNIST 示例
add_executable(example_mnist mnist.cpp)
add_dependencies(example_mnist origindl)
target_link_libraries(example_mnist origindl pthread)
target_include_directories(example_mnist PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 设置输出目录
set_target_properties(example_mnist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/example
)

