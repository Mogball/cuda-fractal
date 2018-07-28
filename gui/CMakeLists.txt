cmake_minimum_required(VERSION 3.1.0)
project(fractal-gui)

# Default to Debug build
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif ()

set(CMAKE_CXX_STANDARD 17)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)

if (Qt5_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_LIST_DIR}/src/*.h)
file(GLOB_RECURSE FORM_FILES   ${CMAKE_CURRENT_LIST_DIR}/src/*.ui)

# Qt MOC and UIC
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

# Main executable
add_executable(fractal-gui ${SOURCE_FILES} ${HEADER_FILES} ${FORM_FILES})

# Link libraries
target_include_directories(fractal-gui PRIVATE
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Gui_INCLUDE_DIRS})

target_link_libraries(fractal-gui
    Qt5::Widgets
    Qt5::Core
    Qt5::Gui)
