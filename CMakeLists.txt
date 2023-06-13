cmake_minimum_required(VERSION 3.20)

project(RTLib CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(OptiX_INSTALL_DIR "C://ProgramData//NVIDIA Corporation//OptiX SDK 7.7.0")

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "-use_fast_math")

message(STATUS CMAKE_CUDA_ARCHITECTURES =${CMAKE_CUDA_ARCHITECTURES})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(RTLIB_TEST_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Data"  CACHE STRING "Data Path For Test")

if (MSVC)
    add_compile_definitions(NOMINMAX)
endif()

add_subdirectory(Ext)
add_subdirectory(Src)
