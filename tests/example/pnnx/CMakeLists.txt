# PNNX 示例
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_executable(example_yolov5 yolov5_test.cpp)
    add_dependencies(example_yolov5 origindl)
    target_link_libraries(example_yolov5 origindl pthread ${OpenCV_LIBS})
    target_include_directories(example_yolov5 PRIVATE ${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(example_yolov5 PRIVATE OPENCV_FOUND)
    
    # 设置输出目录
    set_target_properties(example_yolov5 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/example
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/example
    )
else()
    message(WARNING "OpenCV not found. example_yolov5 will not be built. Install OpenCV to build this demo.")
endif()

