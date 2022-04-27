add_executable(TestEmbree3 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/TestEmbree3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/TestEmbree3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp
)
target_include_directories(TestEmbree3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)

target_link_libraries(TestEmbree3 embree)
set_target_properties(TestEmbree3 PROPERTIES FOLDER RTLib/Test/TestEmbree3)