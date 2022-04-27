if (glad_FOUND)
    if (imgui_FOUND)
		add_library(TestGLFW-GLAD-ImGui-Lib
			${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_GLAD_Imgui.h
			${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_GLAD_Imgui.cpp
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_glfw.h
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_glfw.cpp
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_opengl3.h
			${imgui_INCLUDE_DIR}/../res/bindings/imgui_impl_opengl3.cpp
		)
		target_include_directories(TestGLFW-GLAD-ImGui-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include PUBLIC ${imgui_INCLUDE_DIR}/../res/bindings/)
		target_link_libraries(TestGLFW-GLAD-ImGui-Lib TestGLFW-Common-Lib glad::glad imgui::imgui)
		set_target_properties(TestGLFW-GLAD-ImGui-Lib PROPERTIES FOLDER RTLib/Test/TestGLFW/GLAD_ImGui)
        add_executable(TestGLFW-GLAD-ImGui-Test ${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp)
        target_link_libraries(TestGLFW-GLAD-ImGui-Test PUBLIC TestGLFW-GLAD-ImGui-Lib )
        set_target_properties(TestGLFW-GLAD-ImGui-Test PROPERTIES FOLDER RTLib/Test/TestGLFW/GLAD_ImGui)
	endif()
endif()
