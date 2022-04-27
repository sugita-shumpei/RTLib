add_library(TestGLFW-Common-Lib
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestGLFW_Common.h
	${CMAKE_CURRENT_SOURCE_DIR}/Src/TestGLFW_Common.cpp
)
target_include_directories(TestGLFW-Common-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(TestGLFW-Common-Lib PUBLIC TestLib glfw)
set_target_properties(TestGLFW-Common-Lib PROPERTIES FOLDER RTLib/Test/TestGLFW/Common)
add_executable(TestGLFW-Common-Test
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp
)
target_link_libraries(TestGLFW-Common-Test PUBLIC TestGLFW-Common-Lib)
set_target_properties(TestGLFW-Common-Test PROPERTIES FOLDER RTLib/Test/TestGLFW/Common)
