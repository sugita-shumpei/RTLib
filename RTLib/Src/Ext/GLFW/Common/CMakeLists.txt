set(RTLIB_EXT_GLFW_COMMON_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GLFW/GLFWUtility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GLFW/GLFWCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GLFW/GLFWContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWContext.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GLFW/GLFWWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWWindow.cpp
)

add_library(RTLib-Ext-GLFW-Common 
    ${RTLIB_EXT_GLFW_COMMON_SOURCES}
)
target_include_directories(RTLib-Ext-GLFW-Common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-GLFW-Common PUBLIC RTLib-Core glfw)
set_target_properties(RTLib-Ext-GLFW-Common PROPERTIES FOLDER RTLib/RTLib/Ext/GLFW)
