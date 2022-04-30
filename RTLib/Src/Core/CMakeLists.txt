add_library(RTLib-Core 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/Buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Core/CpuBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/CpuBuffer.cpp
)
target_include_directories(RTLib-Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
set_target_properties(RTLib-Core PROPERTIES FOLDER RTLib/RTLib/Core)