glad_add_library(glad_gl_core_46 STATIC REPRODUCIBLE  MX API gl:core=4.6)
add_library(RTLib-Backends-OGL4-Lib STATIC 
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4Context.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4Buffer.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4Texture.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4VertexArray.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4Framebuffer.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4Renderbuffer.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4Shader.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4Program.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/OGL4/OGL4ProgramPipeline.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4Context.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4Buffer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4Texture.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4VertexArray.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4Framebuffer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4Renderbuffer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4Shader.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4Program.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4ProgramPipeline.cpp
)
target_include_directories(RTLib-Backends-OGL4-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Backends-OGL4-Lib PRIVATE glad_gl_core_46)