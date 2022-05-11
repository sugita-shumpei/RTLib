add_library(RTLib-Ext-GL 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLTexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLContext.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLTexture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLContext.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLResource.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLResource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLBindable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLBindable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLProgramPipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLProgramPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLProgram.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLShader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLQuery.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLQuery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLTexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLTexture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLSampler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLSampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLFramebuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLFramebuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLRenderbuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLRenderbuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLVertexArray.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internal/ImplGLVertexArray.cpp
)
target_include_directories(RTLib-Ext-GL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-GL PUBLIC RTLib-Core glad::glad)
set_target_properties(RTLib-Ext-GL PROPERTIES FOLDER RTLib/RTLib/Ext/GL)

if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Test/shaders")
endif()

set(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Src/TestConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test/TestConfig.h)

add_executable(
    RTLib-Ext-GL-Test
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/TestConfig.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test330.vert 
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test330.frag 
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test410.vert 
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test410.frag  
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test420.vert 
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test420.frag 
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test460.vert 
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test460.frag


)
target_include_directories(RTLib-Ext-GL-Test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)
target_link_libraries(RTLib-Ext-GL-Test PUBLIC RTLib-Ext-GL glfw)
set_target_properties(RTLib-Ext-GL-Test PROPERTIES FOLDER RTLib/RTLib/Ext/GL)

if (Vulkan_FOUND)
    add_custom_command(TARGET RTLib-Ext-GL-Test POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test460.vert ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.vert
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test460.frag ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.frag
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test420.vert ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test420.vert
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test420.frag ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test420.frag
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test410.vert ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test410.vert
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test410.frag ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test410.frag
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test330.vert ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test330.vert
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test330.frag ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test330.frag

        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test460.vert -o ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.vert.spv > ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.vert.log 2>&1
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Src/shaders/Test460.frag -o ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.frag.spv > ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.frag.log 2>&1
    )
endif()
