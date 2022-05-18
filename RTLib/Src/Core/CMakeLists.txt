set(RTLIB_CORE_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/UuidDefinitions.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Context.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Buffer.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Image.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Texture.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Sampler.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Program.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/AccelerationStructure.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/AccelerationStructureBuildInput.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/AccelerationStructureInstance.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/ShaderTable.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/BinaryReader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/BinaryReader.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/BinaryWriter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/BinaryWriter.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/BaseObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/BaseObject.cpp
)

add_library(RTLib-Core ${RTLIB_CORE_SOURCES}
)
target_include_directories(RTLib-Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Core PUBLIC Imath::Imath)
set_target_properties(RTLib-Core PROPERTIES FOLDER RTLib/RTLib/Core)
add_executable(RTLib-Core-Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibCoreTest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibCoreTest.cpp
)

target_include_directories(RTLib-Core-Test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include)
target_link_libraries(RTLib-Core-Test PUBLIC RTLib-Core glfw)

set_target_properties(RTLib-Core-Test PROPERTIES FOLDER RTLib/RTLib/Core)
