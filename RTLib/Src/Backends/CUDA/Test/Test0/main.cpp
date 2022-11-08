#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <cassert>
#include <optional>
#include <unordered_map>

#ifndef NDEBUG
#define RTLIB_BACKENDS_CU_DEBUG_ASSERT(EXPR) \
do { \
	CUresult rtlib_backends_cu_debug_assert_res = EXPR;\
	if (rtlib_backends_cu_debug_assert_res==CUDA_SUCCESS){\
		break; \
	} \
	const char* errStr;\
	cuGetErrorName(rtlib_backends_cu_debug_assert_res, &errStr);\
	std::cerr << "RTLIB_BACKENDS_CU_ERROR: " << errStr << " In File: " __FILE__ << " In Line: " << __LINE__ << " In Call" << #EXPR << std::endl; \
}while(0)
#else
#define RTLIB_BACKENDS_CU_DEBUG_ASSERT(EXPR) (void)EXPR;
#endif

class AkariCudaEntry {

};
class AkariCudaDevice {

};
class AkariCudaContext {

};
class AKariCudaStream {

};
class AkariCudaMemory {
	
};

int main(int argc, const char** argv) {
	RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuInit(0));
	{
		CUdevice  cuDevice;
		RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuDeviceGet(&cuDevice,0));
		CUcontext cuContext1;
		RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuCtxCreate(&cuContext1,0,cuDevice));
		RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuCtxSetCurrent(cuContext1));
		CUstream cuStream;
		CUdeviceptr cuMemory1;
		{
			RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuStreamCreate(&cuStream, CU_STREAM_DEFAULT));
			{
				RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuMemAlloc(&cuMemory1, 1024 * 1024*1024));
				CUcontext cuContext2;
				RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuCtxCreate(&cuContext2, 0, cuDevice));
				RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuMemFreeAsync(cuMemory1,cuStream)); //Error
				RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuCtxSetCurrent(cuContext2));
				RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuStreamSynchronize(cuStream)); //Error
				RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuCtxDestroy(cuContext2));
			}
			RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuStreamDestroy(cuStream));
		}

		RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuCtxDestroy(cuContext1));
		//RTLIB_BACKENDS_CU_DEBUG_ASSERT(cuCtxDestroy(cuContext2));
	}
}