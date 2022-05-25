set(RTLIB_EXT_GL_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLUtility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLTypeFormat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/UuidDefinitions.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLContextImpl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLContextState.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLContextState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLContext.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLUniqueIdentifier.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLObjectBase.h
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLRectRenderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLRectRenderer.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLNatives.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLNatives.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLBuffer.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLTexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLTexture.cpp 
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLImage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLImage.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLShader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLShader.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLProgram.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLProgram.cpp    

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLProgramPipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLProgramPipeline.cpp  

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLVertexArray.h  
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLVertexArray.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLFramebuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLFramebuffer.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLRenderbuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLRenderbuffer.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLMesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLMesh.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLTypeConversions.h
)
add_library(RTLib-Ext-GL 
    ${RTLIB_EXT_GL_SOURCES}
)

target_include_directories(RTLib-Ext-GL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internals)

target_link_libraries(RTLib-Ext-GL PUBLIC RTLib-Core glad::glad)

set_target_properties(RTLib-Ext-GL PROPERTIES FOLDER RTLib/RTLib/Ext/GL)

set(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders)
set(RTLIB_EXT_GL_TEST_CONFIG_DATA_PATH "${RTLIB_TEST_DATA_PATH}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtGLTestConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test/RTLibExtGLTestConfig.h)

add_executable(RTLib-Ext-GL-Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibExtGLTest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtGLTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtGLTestConfig.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test460.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test460.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test420.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test420.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test410.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test410.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test330.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test330.frag
)

target_include_directories(RTLib-Ext-GL-Test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)
target_link_libraries(RTLib-Ext-GL-Test PUBLIC RTLib-Ext-GL glfw stb::stb)

set_target_properties(RTLib-Ext-GL-Test PROPERTIES FOLDER RTLib/RTLib/Ext/GL)
if (Vulkan_FOUND)
    add_custom_command(TARGET RTLib-Ext-GL-Test POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test460.vert ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.vert
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test460.frag ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.frag
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test420.vert ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test420.vert
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test420.frag ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test420.frag
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test410.vert ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test410.vert
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test410.frag ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test410.frag
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test330.vert ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test330.vert
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test330.frag ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test330.frag

        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test460.vert -o ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.vert.spv > ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.vert.log 2>&1
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test460.frag -o ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.frag.spv > ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders/Test460.frag.log 2>&1
    )
endif()
