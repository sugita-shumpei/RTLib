set(RTLIB_EXT_GL_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLUtility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/UuidDefinitions.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/GL/GLContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLContextImpl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/GLContext.cpp

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

add_executable(RTLib-Ext-GL-Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibExtGLTest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtGLTest.cpp
)

target_include_directories(RTLib-Ext-GL-Test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include)
target_link_libraries(RTLib-Ext-GL-Test PUBLIC RTLib-Ext-GL glfw)

set_target_properties(RTLib-Ext-GL-Test PROPERTIES FOLDER RTLib/RTLib/Ext/GL)