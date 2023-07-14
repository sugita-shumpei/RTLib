find_package(glfw3 REQUIRED)

set(SAMPLE_SAMPLE1_DATA_PATH ${CMAKE_SOURCE_DIR}/Data)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Sample1Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/Sample1Config.h)
add_executable(
	RTLib-Samples-Sample1 
	Sample1.h 
	Sample1.cpp 
	 
	 
	 
	
	${CMAKE_CURRENT_BINARY_DIR}/Sample1Config.h
)
target_include_directories(RTLib-Samples-Sample1  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(RTLib-Samples-Sample1 PUBLIC RTLib-Core RTLib-Scene glad_gl_core_46 glfw)
add_custom_command(TARGET RTLib-Samples-Sample1 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:RTLib-Samples-Sample1> $<TARGET_RUNTIME_DLLS:RTLib-Samples-Sample1>
  COMMAND_EXPAND_LISTS
)