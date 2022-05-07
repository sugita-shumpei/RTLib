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

)
target_include_directories(RTLib-Ext-GL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-GL PUBLIC RTLib-Core glad::glad)
set_target_properties(RTLib-Ext-GL PROPERTIES FOLDER RTLib/RTLib/Ext/GL)

add_executable(RTLib-Ext-GL-Test ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test.cpp)
target_link_libraries(RTLib-Ext-GL-Test PUBLIC RTLib-Ext-GL glfw)
set_target_properties(RTLib-Ext-GL-Test PROPERTIES FOLDER RTLib/RTLib/Ext/GL)
