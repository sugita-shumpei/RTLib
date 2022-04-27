cmake_minimum_required(VERSION 3.20)

project(RTLib CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(RTLib)