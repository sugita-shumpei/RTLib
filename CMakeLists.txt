cmake_minimum_required(VERSION 3.20)

project(RTLib CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(RTLIB_TEST_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Data"  CACHE STRING "Data Path For Test")

if (MSVC)
    add_compile_definitions(NOMINMAX )
endif()

add_subdirectory(RTLib)
