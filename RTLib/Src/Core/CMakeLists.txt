set(RTLIB_CORE_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/TypeFormat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Utility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/UuidDefinitions.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Context.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Window.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Keyboard.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Mouse.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Cursor.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Buffer.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Image.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Image.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Texture.cpp

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

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Exceptions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Exceptions.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Camera.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/ShaderTableLayout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/ShaderTableLayout.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Math.h
    
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/AABB.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/VariableMap.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/World.h

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Scene.h
)

add_library(RTLib-Core ${RTLIB_CORE_SOURCES}
)
target_include_directories(RTLib-Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Core PUBLIC Imath::Imath nlohmann_json::nlohmann_json PRIVATE tinyobjloader stb::stb tinyexr::tinyexr glm::glm)
set_target_properties(RTLib-Core PROPERTIES FOLDER RTLib/RTLib/Core)
set(RTLIB_CORE_TEST_CONFIG_DATA_PATH "${RTLIB_TEST_DATA_PATH}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibCoreTestConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test/RTLibCoreTestConfig.h)
add_executable(RTLib-Core-Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include/RTLibCoreTest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/RTLibCoreTestConfig.h.in 
    ${CMAKE_CURRENT_SOURCE_DIR}/Test/Src/RTLibCoreTest.cpp
)

target_include_directories(RTLib-Core-Test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Test/Include PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Test)
target_link_libraries(RTLib-Core-Test PUBLIC RTLib-Core glfw )

set_target_properties(RTLib-Core-Test PROPERTIES FOLDER RTLib/RTLib/Core)
