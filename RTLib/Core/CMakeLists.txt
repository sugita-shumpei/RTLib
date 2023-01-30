find_package(nlohmann_json REQUIRED)
find_package(Imath REQUIRED)

add_library(RTLib-Core STATIC
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTDataType.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTDataType.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTObject.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTObject.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTUuid.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTUuid.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTSession.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTSession.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTActor.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTActor.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTComponent.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTComponent.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTSerialized.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTSerialized.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTTransform.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTTransform.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTVector.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTVector.cpp

)
target_include_directories(RTLib-Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Core PUBLIC Imath::Imath nlohmann_json::nlohmann_json)
set_target_properties(RTLib-Core PROPERTIES FOLDER RTLib/Core)
add_subdirectory(Test)