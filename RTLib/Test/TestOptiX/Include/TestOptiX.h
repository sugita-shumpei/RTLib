#ifndef TEST_TEST_OPTIX_H
#define TEST_TEST_OPTIX_H
#include <TestApplication.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#define TEST_MACRO_CU_CHECK(CALL) \
do { \
	CUresult res = CALL;\
	if (res != CUDA_SUCCESS){\
		std::stringstream ss; \
		ss << "Optix Call (" << #CALL << ") failed with code "<< res << "(line " << __LINE__ << ")\n";\
		throw std::runtime_error(ss.str());\
	}\
}while (0)
#define TEST_MACRO_CUDA_CHECK(CALL) \
do { \
	cudaError_t res = CALL;\
	if (res != cudaSuccess){\
		std::stringstream ss; \
		ss << " CUDA Call (" << #CALL << ") failed with code "<< res << "(line " << __LINE__ << ")\n";\
		throw std::runtime_error(ss.str()); \
	}\
}while (0)
#define TEST_MACRO_OPTIX_CHECK(CALL) \
do { \
	OptixResult res = CALL;\
	if (res != OPTIX_SUCCESS){\
		std::stringstream ss; \
		ss << "Optix Call (" << #CALL << ") failed with code "<< res << "(line " << __LINE__ << ")\n";\
		throw std::runtime_error(ss.str());\
	}\
}while (0)

namespace RTLib {
	namespace Test {
		class TestOptiXApplication : public TestLib::TestApplication{
		public:
			TestOptiXApplication()noexcept;
			virtual ~TestOptiXApplication()noexcept;

		protected:
			virtual void Init()			 override;
			virtual void Main()          override;
			virtual void Free()noexcept  override;
		private:
			CUdevice           m_DevCuda;
			CUcontext		   m_CtxCuda;
			OptixDeviceContext m_CtxOpx7;
		};
	}
}
#endif
