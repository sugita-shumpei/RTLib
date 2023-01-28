add_library(RTLib-Core STATIC
${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTActor.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTActor.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTComponent.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTComponent.cpp

${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTSerialized.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTSerialized.cpp


${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/RTTransform.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/RTTransform.cpp

)
target_include_directories(RTLib-Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
set_target_properties(RTLib-Core PROPERTIES FOLDER RTLib/Core)
add_subdirectory(Test)