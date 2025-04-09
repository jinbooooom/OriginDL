# cmake_minimum_required(VERSION 3.12)
# set(TARGET_NAME test__sum)
# project(ProjectName)

# set(NUMCPP_INCLUDES "../../../3rd/numCpp")

# add_executable(${TARGET_NAME} 
#     main.cpp
# )

# target_include_directories(${TARGET_NAME} PRIVATE 
#     ./
#     ${NUMCPP_INCLUDES}
# )

cmake_minimum_required (VERSION 3.12)

# 获取最后一个目录名，并将其赋值给 PROJECT_NAME 变量
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${BIN_PREFIX}_${PROJECT_NAME})

include_directories(${ORIGINDL_ROOT_DIR}/3rd/numCpp)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ SRC)

add_executable(${PROJECT_NAME} ${SRC}) 
