
if (Vulkan_FOUND)
	add_library(TestGLFW-Vulkan-Lib
		${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_Vulkan.h
		${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_Vulkan.cpp
	)
	target_include_directories(TestGLFW-Vulkan-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
	target_link_libraries(TestGLFW-Vulkan-Lib PUBLIC TestGLFW-Common-Lib TestVulkan-Lib)
	set_target_properties(TestGLFW-Vulkan-Lib PROPERTIES FOLDER Test/TestGLFW/Vulkan)

	set(TEST_GLFW_VULKAN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
		if (NOT EXISTS ${TEST_GLFW_VULKAN_BINARY_DIR}/vk_layer_settings.txt)
		add_custom_command(
		OUTPUT  ${TEST_GLFW_VULKAN_BINARY_DIR}/vk_layer_settings.txt
		COMMAND ${CMAKE_COMMAND} -E copy ${Vulkan_INCLUDE_DIR}/../Config/vk_layer_settings.txt ${TEST_GLFW_VULKAN_BINARY_DIR}/vk_layer_settings.txt
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
	endif()
    add_executable(TestGLFW-Vulkan-Test
        ${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp
		${TEST_GLFW_VULKAN_BINARY_DIR}/vk_layer_settings.txt
    )
    target_link_libraries(TestGLFW-Vulkan-Test TestGLFW-Vulkan-Lib)
    set_target_properties(TestGLFW-Vulkan-Test PROPERTIES FOLDER Test/TestGLFW/Vulkan)
endif()
