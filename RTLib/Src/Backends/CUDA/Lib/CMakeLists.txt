add_library(RTLib-Backends-CUDA-Lib 
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDAEntry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDADevice.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDAContext.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDAStream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDALinearMemory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDAPinnedHostMemory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDAArray.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDAMipmappedArray.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDASurface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDATexture.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDAModule.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/CUDAFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/Math/Preprocessors.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/Math/Math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/Math/Matrix.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/Math/VectorFunctions.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/Math/TypeTraits.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/Math/Random.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Backends/CUDA/Math/Hash.h

  ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internals/CUDAInternals.h

  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAEntry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDADevice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAContext.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAStream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDALinearMemory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAPinnedHostMemory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAArray.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAMipmappedArray.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDASurface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDATexture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAModule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/CUDAFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internals/CUDAInternals.cpp
)
target_include_directories(RTLib-Backends-CUDA-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/Internals)
target_link_libraries(RTLib-Backends-CUDA-Lib PRIVATE CUDA::cuda_driver fmt::fmt)
set(RTLIB_BACKENDS_CUDA_NVCC_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/Include CACHE STRING "RTLib Backends Cuda Include Dirs")
set(RTLIB_BACKENDS_CUDA_NVCC_OPTIONS -arch compute_75 -use_fast_math -lineinfo -rdc true -m64 --std c++17   CACHE STRING "RTLib Backends Cuda Nvcc Options")

