
if (Vulkan_FOUND)
	add_library(TestGLFW-Vulkan-Lib
		${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_Vulkan.h
		${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_Vulkan.cpp
	)
	target_include_directories(TestGLFW-Vulkan-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
	target_link_libraries(TestGLFW-Vulkan-Lib PUBLIC TestGLFW-Common-Lib Vulkan::Vulkan)
	set_target_properties(TestGLFW-Vulkan-Lib PROPERTIES FOLDER RTLib/Test/TestGLFW/Vulkan)
    add_executable(TestGLFW-Vulkan-Test
        ${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp
    )
    target_link_libraries(TestGLFW-Vulkan-Test TestGLFW-Vulkan-Lib)
    set_target_properties(TestGLFW-Vulkan-Test PROPERTIES FOLDER RTLib/Test/TestGLFW/Vulkan)
endif()
