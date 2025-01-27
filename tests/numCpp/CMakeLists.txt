
cmake_minimum_required (VERSION 3.12)
project (numcpp_tests)

set(BIN_PREFIX numcpp)

# 在当前 CMakeLists.txt 文件中使用 file 命令查找所有子目录
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

# 遍历子目录
foreach(child ${children})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
    # 如果子目录存在 CMakeLists.txt，则添加该子目录
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${child}/CMakeLists.txt)
      add_subdirectory(${child})
    endif()
  endif()
endforeach()


