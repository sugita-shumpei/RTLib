#ifndef TEST_TESTLIB_CONTEXT__H
#define TEST_TESTLIB_CONTEXT__H
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <iostream>
#include <optix.h>
#include <cuda.h>
namespace TestLib
{
	struct Context
	{
		 Context();
		~Context();

		Context(const Context&) = delete;
		Context(Context&&) = delete;

		Context& operator=(const Context&) = delete;
		Context& operator=(Context&&) = delete;

		void init();
		void free();

		auto get_cuda_context() const noexcept -> CUcontext;

		auto get_cuda_device() const noexcept  -> CUdevice;

		auto get_opx7_device_context() const noexcept -> OptixDeviceContext;

	private:
		void init_cuda_driver();
		void free_cuda_driver();

		void init_opx7_device_context();
		void free_opx7_device_context();

		static void opx7_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata)
		{
			constexpr const char* level2Str[] = { "Disable","Fatal","Error","Warning","Print" };
			printf("[%s][%s]: %s\n", level2Str[level], tag, message);
		}
	private:
		CUcontext          m_CudaContext;
		CUdevice           m_CudaDevice;
		OptixDeviceContext m_Opx7DeviceContext;
	};
}
#endif
