if (glad_FOUND)
	add_library(TestGLFW-GLAD-Lib
		${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_GLAD.h
		${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_GLAD.cpp
	)
	target_include_directories(TestGLFW-GLAD-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
	target_link_libraries(TestGLFW-GLAD-Lib TestGLFW-Common-Lib RTLib-Ext-GL glad::glad)
	set_target_properties(TestGLFW-GLAD-Lib PROPERTIES FOLDER Test/TestGLFW/GLAD)
	add_executable(TestGLFW-GLAD-Test
		${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp
	)
	target_link_libraries(TestGLFW-GLAD-Test PUBLIC TestGLFW-GLAD-Lib)
	set_target_properties(TestGLFW-GLAD-Test PROPERTIES FOLDER Test/TestGLFW/GLAD)
endif()
