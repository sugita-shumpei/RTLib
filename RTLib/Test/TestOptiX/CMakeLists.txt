add_executable(TestOptiX 
${CMAKE_CURRENT_SOURCE_DIR}/Include/TestOptiX.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/TestOptiX.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp
)
target_include_directories(TestOptiX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(TestOptiX TestLib OptiX::OptiX CUDA::cudart CUDA::cuda_driver)
set_target_properties(TestOptiX PROPERTIES FOLDER RTLib/Test/TestOptiX)
