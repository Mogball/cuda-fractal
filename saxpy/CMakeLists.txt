cmake_minimum_required(VERSION 3.1)
project(saxpy CUDA)
add_executable(saxpy saxpy.cu)
target_include_directories(saxpy PRIVATE ${INCLUDE_DIR})
