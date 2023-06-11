find_package(OptiX REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_library(OTK-Error INTERFACE)

target_sources(OTK-Error INTERFACE 
	# ${CMAKE_CURRENT_SOURCE_DIR}/.. = Ext/CMake/opitx-toolkit
	# ${CMAKE_CURRENT_SOURCE_DIR}/../../.. = Ext

	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Error/cudaErrorCheck.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Error/cuErrorCheck.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Error/ErrorCheck.h
)
target_include_directories(OTK-Error INTERFACE ${OptiXToolkit_ROOT}/Memory/include)
target_link_libraries(OTK-Error INTERFACE OptiX::OptiX CUDA::cuda_driver CUDA::cudart)

add_library(OTK-Memory INTERFACE)
target_sources(OTK-Memory INTERFACE 
	# ${CMAKE_CURRENT_SOURCE_DIR}/.. = Ext/CMake/opitx-toolkit
	# ${CMAKE_CURRENT_SOURCE_DIR}/../../.. = Ext

	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/Allocators.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/Assert.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/BinnedSuballocator.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/BitCast.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/CudaCheck.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/DeviceBuffer.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/DeviceFixedPool.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/DeviceRingBuffer.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/FixedSuballocator.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/HeapSuballocator.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/MemoryBlockDesc.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/MemoryPool.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/RingSuballocator.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/Memory/SyncVector.h
)
target_include_directories(OTK-Memory INTERFACE ${OptiXToolkit_ROOT}/Memory/include)
target_link_libraries(OTK-Memory INTERFACE OptiX::OptiX CUDA::cuda_driver CUDA::cudart)

add_library(OTK-OptiXMemory INTERFACE)
target_sources(OTK-OptiXMemory INTERFACE 
	# ${CMAKE_CURRENT_SOURCE_DIR}/.. = Ext/CMake/opitx-toolkit
	# ${CMAKE_CURRENT_SOURCE_DIR}/../../.. = Ext

	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/OptiXMemory/Builders.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/OptiXMemory/Record.h
	${OptiXToolkit_ROOT}/Memory/include/OptiXToolkit/OptiXMemory/SyncRecord.h
)
target_include_directories(OTK-OptiXMemory INTERFACE ${OptiXToolkit_ROOT}/Memory/include)
target_link_libraries(OTK-OptiXMemory INTERFACE OTK-Memory OTK-Error OptiX::OptiX CUDA::cuda_driver CUDA::cudart)