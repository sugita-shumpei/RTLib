add_executable(RTLib-Backends-CUDA-Test-Test0 ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(RTLib-Backends-CUDA-Test-Test0 PUBLIC RTLib-Backends-CUDA-Lib CUDA::cuda_driver)

