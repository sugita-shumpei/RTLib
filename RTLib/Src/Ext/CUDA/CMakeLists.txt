set(RTLIB_EXT_CUDA_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAUtility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDATypeFormat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDACommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/UuidDefinitions.h


    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAContext.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDABuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDABuffer.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDATexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDATexture.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAImage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAImage.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAProgram.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAProgram.cpp  

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAModule.h  
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAModule.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAFunction.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAFunction.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/CUDAStream.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAStream.cpp 

    
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDATypeConversions.h 
)

add_library(RTLib-Ext-CUDA 
    ${RTLIB_EXT_CUDA_SOURCES}
)
target_include_directories(RTLib-Ext-CUDA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-CUDA PUBLIC RTLib-Core CUDA::cuda_driver)
set_target_properties(RTLib-Ext-CUDA PROPERTIES FOLDER RTLib/RTLib/Ext/CUDA)

set(RTLIB_EXT_CUDA_TEST_DATA_PATH "${RTLIB_TEST_DATA_PATH}")
set(RTLIB_EXT_CUDA_TEST_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/Test/cuda")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtCUDATestConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test/RTLibExtCUDATestConfig.h)

add_executable(RTLib-Ext-CUDA-Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtCUDATestConfig.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibExtCUDATest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtCUDATest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/simpleKernel.cu
)
target_include_directories(RTLib-Ext-CUDA-Test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)
target_link_libraries(RTLib-Ext-CUDA-Test PUBLIC RTLib-Ext-CUDA stb::stb)

add_custom_command(TARGET RTLib-Ext-CUDA-Test POST_BUILD 
COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/simpleKernel.cu -arch compute_75 -use_fast_math -lineinfo -rdc true -m64 --std c++17 -ptx -I${CMAKE_CURRENT_SOURCE_DIR}/Include/ -o  ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/simpleKernel.ptx
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/simpleKernel.cu  ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/simpleKernel.cu
)


set_target_properties(RTLib-Ext-CUDA-Test PROPERTIES FOLDER RTLib/RTLib/Ext/CUDA)