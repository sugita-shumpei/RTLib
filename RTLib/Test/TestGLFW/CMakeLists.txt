add_library(TestGLFW 
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW.h
	${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW.cpp
)
target_include_directories(TestGLFW PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(TestGLFW PUBLIC TestLib glfw)
set_target_properties(TestGLFW PROPERTIES FOLDER RTLib/Test/TestGLFW)

if (glad_FOUND)
	add_executable(TestGLFW-GLAD
		${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_GLAD.h
		${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_GLAD.cpp
	)
	target_include_directories(TestGLFW-GLAD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
	target_link_libraries(TestGLFW-GLAD TestGLFW glad::glad)
	set_target_properties(TestGLFW-GLAD PROPERTIES FOLDER RTLib/Test/TestGLFW)
endif()

if (Vulkan_FOUND)
	add_executable(TestGLFW-Vulkan
		${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_Vulkan.h
		${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_Vulkan.cpp
	)
	target_include_directories(TestGLFW-Vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
	target_link_libraries(TestGLFW-Vulkan TestGLFW Vulkan::Vulkan)
	set_target_properties(TestGLFW-Vulkan PROPERTIES FOLDER RTLib/Test/TestGLFW)
endif()