# ResNet 示例
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    
    add_executable(example_resnet resnet_infer.cpp)
    add_dependencies(example_resnet origindl)
    target_link_libraries(example_resnet origindl pthread ${OpenCV_LIBS})
    target_include_directories(example_resnet PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(example_resnet PRIVATE OPENCV_FOUND)
    
    # 设置输出目录
    set_target_properties(example_resnet PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/example
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/example
    )
else()
    message(WARNING "OpenCV not found. example_resnet will not be built. Install OpenCV to build this demo.")
endif()

