set(RTLIB_EXT_CUDA_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAUtility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDACommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAContext.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDABuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDABuffer.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDATexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDATexture.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAArray.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAArray.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAProgram.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAProgram.cpp  

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAModule.h  
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAModule.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAFunction.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAFunction.cpp 
)

add_library(RTLib-Ext-CUDA 
    ${RTLIB_EXT_CUDA_SOURCES}
)
target_include_directories(RTLib-Ext-CUDA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-CUDA PUBLIC RTLib-Core CUDA::cudart CUDA::cuda_driver)
set_target_properties(RTLib-Ext-CUDA PROPERTIES FOLDER RTLib/RTLib/Ext/CUDA)
