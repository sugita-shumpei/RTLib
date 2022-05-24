set(RTLIB_EXT_CUGL_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUGL/CUGLUtility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUGL/CUGLCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUGL/CUGLContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUGLContext.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUGL/CUGLBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUGLBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUGL/CUGLImage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUGLImage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUGL/UuidDefinitions.h
)

add_library(RTLib-Ext-CUGL 
    ${RTLIB_EXT_CUGL_SOURCES}
)
target_include_directories(RTLib-Ext-CUGL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-CUGL PUBLIC RTLib-Ext-CUDA RTLib-Ext-GL)
set_target_properties(RTLib-Ext-CUGL PROPERTIES FOLDER RTLib/RTLib/Ext/CUGL)
