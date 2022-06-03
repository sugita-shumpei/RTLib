set(RTLIB_EXT_OPX7_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Utility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/UuidDefinitions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7ShaderRecord.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Context.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Payload.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Natives.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Natives.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Exceptions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Exceptions.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Module.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Module.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7ProgramGroup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7ProgramGroup.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Pipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Pipeline.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7AccelerationStructure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7AccelerationStructure.cpp  

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7AccelerationStructureInstance.h  
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7AccelerationStructureInstance.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7AccelerationStructureBuildInput.h  
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7AccelerationStructureBuildInput.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7ShaderTableLayout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7ShaderTableLayout.cpp 
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7ShaderTable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7ShaderTable.cpp 
)


add_library(RTLib-Ext-OPX7 
    ${RTLIB_EXT_OPX7_SOURCES}
)
target_include_directories(RTLib-Ext-OPX7 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-OPX7 PUBLIC RTLib-Ext-CUGL OptiX::OptiX tinyobjloader glfw)
set_target_properties(RTLib-Ext-OPX7 PROPERTIES FOLDER RTLib/RTLib/Ext/OPX7)

set(RTLIB_EXT_OPX7_TEST_DATA_PATH "${RTLIB_TEST_DATA_PATH}")
set(RTLIB_EXT_OPX7_TEST_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/Test/cuda")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtOPX7TestConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test/RTLibExtOPX7TestConfig.h)

set(RTLIB_EXT_OPX7_TEST_NVCC_INCLUDE_DIRS ${OptiX_INCLUDE_DIR}  ${RTLIB_EXT_CUDA_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda ${CMAKE_CURRENT_SOURCE_DIR}/Include)
set(RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS -arch compute_75 -use_fast_math -lineinfo -rdc true -m64 --std c++17 -ptx)
foreach(INCLUDE_DIR ${RTLIB_EXT_OPX7_TEST_NVCC_INCLUDE_DIRS})
    list(APPEND RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS -I${INCLUDE_DIR})
endforeach()

add_executable(RTLib-Ext-OPX7-Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtOPX7TestConfig.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibExtOPX7Test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtOPX7Test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleKernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleKernel.cu
)
target_include_directories(RTLib-Ext-OPX7-Test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)
target_link_libraries(RTLib-Ext-OPX7-Test PUBLIC RTLib-Ext-OPX7 stb::stb)
add_custom_command(TARGET RTLib-Ext-OPX7-Test PRE_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleKernel.cu ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleKernel.cu
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleKernel.h  ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleKernel.h)
add_custom_command(TARGET RTLib-Ext-OPX7-Test POST_BUILD 
COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleKernel.cu ${RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS} -o  ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleKernel.ptx
)

set_target_properties(RTLib-Ext-OPX7-Test PROPERTIES FOLDER RTLib/RTLib/Ext/OPX7)