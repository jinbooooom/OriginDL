# MNIST MLP 示例
add_executable(example_mlp_mnist mlp_mnist.cpp)
add_dependencies(example_mlp_mnist origindl)
target_link_libraries(example_mlp_mnist origindl pthread)
target_include_directories(example_mlp_mnist PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 设置输出目录
set_target_properties(example_mlp_mnist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/example
)

# MNIST 卷积网络示例
add_executable(example_conv_mnist conv_mnist.cpp)
add_dependencies(example_conv_mnist origindl)
target_link_libraries(example_conv_mnist origindl pthread)
target_include_directories(example_conv_mnist PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 设置输出目录
set_target_properties(example_conv_mnist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/example
)

