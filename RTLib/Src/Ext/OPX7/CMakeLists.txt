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

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7ShaderTableLayout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7ShaderTableLayout.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7ShaderTable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7ShaderTable.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/Utils/OPX7UtilsPathGuiding.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/Utils/OPX7UtilsMorton.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Utils/OPX7UtilsMorton.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/Utils/OPX7UtilsGrid.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/Utils/OPX7UtilsMortonBuild.cu

    ${CMAKE_CURRENT_BINARY_DIR}/OPX7UtilsMortonBuild.ptx.h
)

add_library(RTLib-Ext-OPX7
    ${RTLIB_EXT_OPX7_SOURCES}
)
add_custom_command(TARGET RTLib-Ext-OPX7 PRE_BUILD
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/Utils/OPX7UtilsMortonBuild.cu -arch compute_75 -use_fast_math -lineinfo -rdc true -m64 --std c++17 -ptx -o ${CMAKE_CURRENT_BINARY_DIR}/OPX7UtilsMortonBuild.ptx
    COMMAND ${CUDAToolkit_BIN_DIR}\\bin2c --const -st --name OPX7UtilsMortonBuildPtxBinary ${CMAKE_CURRENT_BINARY_DIR}/OPX7UtilsMortonBuild.ptx > ${CMAKE_CURRENT_BINARY_DIR}/OPX7UtilsMortonBuild.ptx.h
)

target_include_directories(RTLib-Ext-OPX7 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(RTLib-Ext-OPX7 PUBLIC RTLib-Ext-CUGL OptiX::OptiX RTLib-Ext-GLFW-GL)
set_target_properties(RTLib-Ext-OPX7 PROPERTIES FOLDER RTLib/RTLib/Ext/OPX7)

set(RTLIB_EXT_OPX7_TEST_DATA_PATH "${RTLIB_TEST_DATA_PATH}")
set(RTLIB_EXT_OPX7_TEST_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/Test/cuda")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtOPX7TestConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test/RTLibExtOPX7TestConfig.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/PathGuidingConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/PathGuidingConfig.h)

set(RTLIB_EXT_OPX7_TEST_NVCC_INCLUDE_DIRS ${OptiX_INCLUDE_DIR} ${RTLIB_EXT_CUDA_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda ${CMAKE_CURRENT_SOURCE_DIR}/Include)
set(RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS -arch compute_75 -use_fast_math -lineinfo -rdc true -m64 --std c++17)

foreach(INCLUDE_DIR ${RTLIB_EXT_OPX7_TEST_NVCC_INCLUDE_DIRS})
    list(APPEND RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS -I${INCLUDE_DIR})
endforeach()

add_library(RTLib-Ext-OPX7-Test-Lib
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibExtOPX7TestConfig.h.in

    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibExtOPX7TestApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtOPX7TestApplication.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibExtOPX7Test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtOPX7Test.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleKernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleTrace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleGuide.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleGuide2.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/PathGuidingConfig.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/MortonTreeConfig.h.in
)
set(RTLIB_EXT_OPX7_TEST_PATH_GUIDING_TRACE_VERTEX_SIZE 10)
set(RTLIB_EXT_OPX7_TEST_PATH_GUIDING_STREE_STACK_DEPTH 69)
set(RTLIB_EXT_OPX7_TEST_PATH_GUIDING_DTREE_STACK_DEPTH 24)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/PathGuidingConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/PathGuidingConfig.h
)
set(RTLIB_EXT_OPX7_TEST_MORTON_TREE_TRACE_VERTEX_SIZE 10)
set(RTLIB_EXT_OPX7_TEST_MORTON_TREE_MAX_LEVEL         6)
set(RTLIB_EXT_OPX7_TEST_HASH_GRID_BLOCK_SIZE          32)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/MortonTreeConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/MortonTreeConfig.h
)
target_include_directories(RTLib-Ext-OPX7-Test-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test )

target_link_libraries(RTLib-Ext-OPX7-Test-Lib PUBLIC RTLib-Ext-OPX7 tinyexr::tinyexr stb::stb RTLib-Utils)

add_custom_target(RTLib-Ext-OPX7-Test-Lib-Build-Kernel)
add_custom_command(TARGET RTLib-Ext-OPX7-Test-Lib-Build-Kernel PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleKernel.h  ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleKernel.h
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleTrace.cu  ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleTrace.cu
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleGuide.cu  ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleGuide.cu
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleGuide2.cu ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleGuide2.cu
)
add_custom_command(TARGET RTLib-Ext-OPX7-Test-Lib-Build-Kernel POST_BUILD
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}           ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleTrace.cu ${RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS} -ptx -o ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleTrace.ptx
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -optix-ir ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleTrace.cu ${RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS}      -o ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleTrace.optixir

    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}           ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleGuide.cu ${RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS} -ptx -o ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleGuide.ptx
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -optix-ir ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleGuide.cu ${RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS}      -o ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleGuide.optixir

    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}           ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleGuide2.cu ${RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS} -ptx -o ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleGuide2.ptx
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -optix-ir ${CMAKE_CURRENT_SOURCE_DIR}/Test/cuda/SimpleGuide2.cu ${RTLIB_EXT_OPX7_TEST_NVCC_OPTIONS}      -o ${CMAKE_CURRENT_BINARY_DIR}/Test/cuda/SimpleGuide2.optixir
)

add_dependencies(RTLib-Ext-OPX7-Test-Lib RTLib-Ext-OPX7-Test-Lib-Build-Kernel)

set_target_properties(RTLib-Ext-OPX7-Test-Lib RTLib-Ext-OPX7-Test-Lib-Build-Kernel PROPERTIES FOLDER RTLib/RTLib/Ext/OPX7)

add_executable(RTLib-Ext-OPX7-Test-Main
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtOPX7Main.cpp
)
target_link_libraries(RTLib-Ext-OPX7-Test-Main PUBLIC RTLib-Ext-OPX7-Test-Lib)

target_include_directories(RTLib-Ext-OPX7-Test-Main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)

set_target_properties(RTLib-Ext-OPX7-Test-Main PROPERTIES FOLDER RTLib/RTLib/Ext/OPX7)

add_executable(RTLib-Ext-OPX7-Test-Comp
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtOPX7Comp.cpp
)
target_link_libraries(RTLib-Ext-OPX7-Test-Comp PUBLIC RTLib-Ext-OPX7-Test-Lib)

target_include_directories(RTLib-Ext-OPX7-Test-Comp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)

set_target_properties(RTLib-Ext-OPX7-Test-Comp  PROPERTIES FOLDER RTLib/RTLib/Ext/OPX7)

add_executable(RTLib-Ext-OPX7-Test-ImageSelector
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibExtOPX7ImageSelector.cpp
)
target_link_libraries(RTLib-Ext-OPX7-Test-ImageSelector PUBLIC RTLib-Ext-GLFW-GL tinyexr::tinyexr stb::stb RTLib-Utils)

target_include_directories(RTLib-Ext-OPX7-Test-ImageSelector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)

set_target_properties(RTLib-Ext-OPX7-Test-ImageSelector PROPERTIES FOLDER RTLib/RTLib/Ext/OPX7)