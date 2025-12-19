# MNIST 示例
add_executable(mnist_example mnist.cpp)
add_dependencies(mnist_example origindl)
target_link_libraries(mnist_example origindl pthread)
target_include_directories(mnist_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 设置输出目录
set_target_properties(mnist_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/example
)

