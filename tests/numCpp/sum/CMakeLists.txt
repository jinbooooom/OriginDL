cmake_minimum_required(VERSION 3.22)
set(TARGET_NAME test_sum)
project(ProjectName)

set(NUMCPP_INCLUDES "../../../3rd/numCpp")

add_executable(${TARGET_NAME} 
    main.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE 
    ./
    ${NUMCPP_INCLUDES}
)

