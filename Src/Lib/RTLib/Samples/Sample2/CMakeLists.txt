find_package(glfw3 REQUIRED)

set(SAMPLE_SAMPLE2_DATA_PATH ${CMAKE_SOURCE_DIR}/Data)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Sample2Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/Sample2Config.h)
add_executable(
	RTLib-Samples-Sample2 
	Sample2.h 
	Sample2.cpp 
	 
	${CMAKE_CURRENT_BINARY_DIR}/Sample2Config.h
)
target_include_directories(RTLib-Samples-Sample2  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(RTLib-Samples-Sample2 PUBLIC RTLib-Core RTLib-Scene glad_gl_core_46 glfw)
add_custom_command(TARGET RTLib-Samples-Sample2 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:RTLib-Samples-Sample2> $<TARGET_RUNTIME_DLLS:RTLib-Samples-Sample2>
  COMMAND_EXPAND_LISTS
)