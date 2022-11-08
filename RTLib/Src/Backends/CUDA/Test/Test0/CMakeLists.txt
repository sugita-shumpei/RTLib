add_executable(RTLib-CUDA-Backends-Test-Test0 ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(RTLib-CUDA-Backends-Test-Test0 PUBLIC CUDA::cuda_driver)