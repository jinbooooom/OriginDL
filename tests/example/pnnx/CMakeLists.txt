# PNNX 示例
add_executable(example_yolov5 yolov5_test.cpp)
add_dependencies(example_yolov5 origindl)
target_link_libraries(example_yolov5 origindl pthread)
target_include_directories(example_yolov5 PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 设置输出目录
set_target_properties(example_yolov5 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/example
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/example
)

