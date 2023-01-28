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
	if (imgui_FOUND)
		add_executable(TestGLFW-Imgui-GLAD
			${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_Imgui_GLAD.h
			${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_Imgui_GLAD.cpp
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_glfw.h
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_glfw.cpp
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_opengl3.h
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_opengl3.cpp
		)
		target_include_directories(TestGLFW-Imgui-GLAD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include PRIVATE ${imgui_INCLUDE_DIR}/../res/bindings/)
		target_link_libraries(TestGLFW-Imgui-GLAD TestGLFW glad::glad imgui::imgui)
		set_target_properties(TestGLFW-Imgui-GLAD PROPERTIES FOLDER RTLib/Test/TestGLFW)
	endif()
endif()

if (Vulkan_FOUND)
	add_executable(TestGLFW-Vulkan
		${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_Vulkan.h
		${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_Vulkan.cpp
	)
	target_include_directories(TestGLFW-Vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
	target_link_libraries(TestGLFW-Vulkan TestGLFW Vulkan::Vulkan)
	set_target_properties(TestGLFW-Vulkan PROPERTIES FOLDER RTLib/Test/TestGLFW)
	if (imgui_FOUND)
		add_executable(TestGLFW-Imgui-Vulkan
			${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_Imgui_Vulkan.h
			${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_Imgui_Vulkan.cpp
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_glfw.h
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_glfw.cpp
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_vulkan.h
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_vulkan.cpp
		)
		target_include_directories(TestGLFW-Imgui-Vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include PRIVATE ${imgui_INCLUDE_DIR}/../res/bindings/)
		target_link_libraries(TestGLFW-Imgui-Vulkan TestGLFW Vulkan::Vulkan imgui::imgui)
		set_target_properties(TestGLFW-Imgui-Vulkan PROPERTIES FOLDER RTLib/Test/TestGLFW)
	endif()
endif()