set(RTLIB_EXT_GLFW_VK_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GLFW/VK/GLFWVulkanWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWVulkanWindow.cpp
)
add_library(RTLib-Ext-GLFW-VK 
${RTLIB_EXT_GLFW_VK_SOURCES}
)
target_include_directories(RTLib-Ext-GLFW-VK PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-GLFW-VK PUBLIC RTLib-Ext-GLFW-Common)
set_target_properties(RTLib-Ext-GLFW-VK PROPERTIES FOLDER RTLib/RTLib/Ext/GLFW)
