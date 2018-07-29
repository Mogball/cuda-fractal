cmake_minimum_required(VERSION 3.1)
project(stride CUDA)
add_executable(stride stride.cu)
target_include_directories(stride PRIVATE ${INCLUDE_DIR})
