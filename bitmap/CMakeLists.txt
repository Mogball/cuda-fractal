cmake_minimum_required(VERSION 3.1.0)
project(bitmap)

add_library(bitmap bitmap.cpp)
set_property(TARGET bitmap PROPERTY CXX_STANDARD 11)

add_executable(bitmap_test bitmap_test.cpp)
set_property(TARGET bitmap_test PROPERTY CXX_STANDARD 11)

target_include_directories(bitmap_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)
target_link_libraries(bitmap_test bitmap)
