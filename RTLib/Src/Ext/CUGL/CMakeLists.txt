set(RTLIB_EXT_CUGL_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUGL/CUGLUtility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUGL/CUGLCommon.h

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

set(RTLIB_EXT_CUGL_TEST_CONFIG_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders)
set(RTLIB_EXT_CUGL_TEST_CONFIG_DATA_PATH "${RTLIB_TEST_DATA_PATH}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtCUGLTestConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test/RTLibExtCUGLTestConfig.h)

add_executable(RTLib-Ext-CUGL-Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibExtCUGLTest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtCUGLTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtCUGLTestConfig.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/simpleKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test460.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test460.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test420.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test420.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test410.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test410.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test330.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/shaders/Test330.frag
)
target_include_directories(RTLib-Ext-CUGL-Test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)
target_link_libraries(RTLib-Ext-CUGL-Test PUBLIC RTLib-Ext-CUGL glfw stb::stb)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Test/shaders)
set(RTLIB_EXT_CUGL_TEST_NVCC_INCLUDE_DIRS ${OptiX_INCLUDE_DIR} ${RTLIB_EXT_CUDA_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda)
set(RTLIB_EXT_CUGL_TEST_NVCC_OPTIONS -arch compute_75 -use_fast_math -lineinfo -rdc true -m64 --std c++17 -ptx)
foreach(INCLUDE_DIR ${RTLIB_EXT_CUGL_TEST_NVCC_INCLUDE_DIRS})
    list(APPEND RTLIB_EXT_CUGL_TEST_NVCC_OPTIONS -I${INCLUDE_DIR})
endforeach()

set_target_properties(RTLib-Ext-CUGL-Test PROPERTIES FOLDER RTLib/RTLib/Ext/CUGL)
if (Vulkan_FOUND)
    add_custom_command(TARGET RTLib-Ext-CUGL-Test POST_BUILD 
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
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/simpleKernel.cu ${RTLIB_EXT_CUGL_TEST_NVCC_OPTIONS} -o ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/simpleKernel.ptx
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/simpleKernel.cu  ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/simpleKernel.cu
    )
endif()
