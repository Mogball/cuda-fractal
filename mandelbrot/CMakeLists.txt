cmake_minimum_required(VERSION 3.1)
project(mandelbrot CXX CUDA)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -O3)
add_library(gpu mandelbrot.cu)
set_property(TARGET gpu PROPERTY CUDA_STANDARD 11)
target_include_directories(gpu PRIVATE ${INCLUDE_DIR})

add_executable(gpu_test main.cpp)
set_property(TARGET gpu_test PROPERTY CXX_STANDARD 11)
target_include_directories(gpu_test PRIVATE ${INCLUDE_DIR})
target_link_libraries(gpu_test gpu bitmap)
