add_executable(RTLib-Test-Core-CpuBuffer 
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/TestCpuBuffer.cpp
)
target_include_directories(RTLib-Test-Core-CpuBuffer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Test-Core-CpuBuffer RTLib-Core)
set_target_properties(RTLib-Test-Core-CpuBuffer PROPERTIES FOLDER RTLib/Test/Core)