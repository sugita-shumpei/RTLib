add_library(RTLib-Backends-GLFW-Lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/GLFW/GLFWEntry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/GLFW/Window/GLFWWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/GLFW/Inputs/GLFWKeyboard.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/GLFW/Inputs/GLFWMouse.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/GLFW/Inputs/GLFWCursor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internals/GLFWInternals.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWEntry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWKeyboard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWMouse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWCursor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFWInternals.cpp
)
target_include_directories(RTLib-Backends-GLFW-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Internals)
target_link_libraries(RTLib-Backends-GLFW-Lib PUBLIC RTLib-Window-Lib RTLib-Inputs-Lib PRIVATE glfw)