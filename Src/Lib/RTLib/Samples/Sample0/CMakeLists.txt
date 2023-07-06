find_package(glfw3 REQUIRED)

set(SAMPLE_SAMPLE0_DATA_PATH ${CMAKE_SOURCE_DIR}/Data)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Sample0Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/Sample0Config.h)
add_executable(RTLib-Samples-Sample0 Sample0.cpp ${CMAKE_CURRENT_BINARY_DIR}/Sample0Config.h)
target_include_directories(RTLib-Samples-Sample0  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(RTLib-Samples-Sample0 PUBLIC RTLib-Core RTLib-Scene glad_gl_core_46 glfw)
add_custom_command(TARGET RTLib-Samples-Sample0 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:RTLib-Samples-Sample0> $<TARGET_RUNTIME_DLLS:RTLib-Samples-Sample0>
  COMMAND_EXPAND_LISTS
)