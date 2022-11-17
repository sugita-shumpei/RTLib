add_executable(RTLib-Backends-OGL4-Test-Test0 ${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp)
target_link_libraries(RTLib-Backends-OGL4-Test-Test0 PRIVATE glad_gl_core_46 glfw)