add_executable(RTLib-Core-Test-Test0 ${CMAKE_CURRENT_SOURCE_DIR}/Include/Test0.h ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test0.cpp)
target_include_directories(RTLib-Core-Test-Test0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Core-Test-Test0 PUBLIC RTLib-Core)
set_target_properties(RTLib-Core-Test-Test0 PROPERTIES FOLDER RTLib/Core/Test/Test0)