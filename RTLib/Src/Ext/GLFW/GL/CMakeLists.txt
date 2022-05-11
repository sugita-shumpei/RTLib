set(RTLIB_EXT_GLFW_GL_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GLFW/GL/GLFWOpenGLWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWOpenGLWindow.cpp
)
add_library(RTLib-Ext-GLFW-GL 
    ${RTLIB_EXT_GLFW_GL_SOURCES}
)
target_include_directories(RTLib-Ext-GLFW-GL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-GLFW-GL PUBLIC RTLib-Ext-GLFW-Common)
set_target_properties(RTLib-Ext-GLFW-GL PROPERTIES FOLDER RTLib/RTLib/Ext/GLFW)
