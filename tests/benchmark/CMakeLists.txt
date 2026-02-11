cmake_minimum_required(VERSION 3.10)

# 设置包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 设置可执行文件输出目录
set(BENCHMARK_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/benchmark)

# 辅助函数：添加单个benchmark可执行文件
function(add_benchmark_executable category operator)
    set(executable_name "bench_${operator}")
    set(source_file "${CMAKE_SOURCE_DIR}/tests/benchmark/${category}/bench_${operator}.cpp")
    
    # 检查源文件是否存在
    if(NOT EXISTS ${source_file})
        return()
    endif()
    
    add_executable(${executable_name} 
        ${source_file} 
        ${CMAKE_SOURCE_DIR}/tests/benchmark/common/benchmark_framework.cpp
    )
    
    add_dependencies(${executable_name} origindl)
    target_link_libraries(${executable_name} origindl pthread)
    
    # Origin后端：如果启用CUDA，添加CUDA相关链接
    if(ENABLE_CUDA AND MAT_BACKEND STREQUAL "ORIGIN")
        target_link_libraries(${executable_name} originmat_cuda ${CUDA_LIBRARIES})
    endif()
    
    target_include_directories(${executable_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    
    set_target_properties(${executable_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BENCHMARK_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BENCHMARK_OUTPUT_DIR}
    )
endfunction()

# 定义各类别算子列表
set(MATH_OPERATORS add sub mul div exp log neg square mat_mul pow sum)
set(SHAPE_OPERATORS transpose reshape cat flatten)
set(ACTIVATION_OPERATORS relu sigmoid softmax silu)
set(CONV_OPERATORS conv2d)
set(POOLING_OPERATORS max_pool2d avg_pool2d adaptive_avg_pool2d)
set(NORMALIZATION_OPERATORS batch_norm)
set(LOSS_OPERATORS softmax_cross_entropy)
set(NN_OPERATORS dropout identity upsample)
set(CUSTOM_OPERATORS linear)

# 为每个类别和算子添加可执行文件
foreach(operator ${MATH_OPERATORS})
    add_benchmark_executable(math ${operator})
endforeach()

foreach(operator ${SHAPE_OPERATORS})
    add_benchmark_executable(shape ${operator})
endforeach()

foreach(operator ${ACTIVATION_OPERATORS})
    add_benchmark_executable(activation ${operator})
endforeach()

foreach(operator ${CONV_OPERATORS})
    add_benchmark_executable(conv ${operator})
endforeach()

foreach(operator ${POOLING_OPERATORS})
    add_benchmark_executable(pooling ${operator})
endforeach()

foreach(operator ${NORMALIZATION_OPERATORS})
    add_benchmark_executable(normalization ${operator})
endforeach()

foreach(operator ${LOSS_OPERATORS})
    add_benchmark_executable(loss ${operator})
endforeach()

foreach(operator ${NN_OPERATORS})
    add_benchmark_executable(nn ${operator})
endforeach()

foreach(operator ${CUSTOM_OPERATORS})
    add_benchmark_executable(custom ${operator})
endforeach()

message(STATUS "Benchmark tests configuration completed")
