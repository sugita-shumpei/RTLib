#ifndef RTLIB_EXT_CUDA_CUDA_FUNCTION_H
#define RTLIB_EXT_CUDA_CUDA_FUNCTION_H
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <RTLib/Ext/CUDA/CUDAStream.h>
#include <vector>
#include <iostream>
#include <string>
namespace RTLib
{
	namespace Ext
	{
		namespace CUDA
		{
			class CUDAModule;
			class CUDANatives;
			class CUDAFunction
			{
			public:
				friend class CUDANatives;
				static auto Load(CUDAModule* cuModule, const char* entryPoint)->CUDAFunction*;
				virtual ~CUDAFunction()noexcept;

				void Destory()noexcept;
				bool Launch(const CUDAKernelLaunchDesc& desc);
			private:
				CUDAFunction(CUDAModule* module, CUfunction function)noexcept;
				auto GetCUfunction()const noexcept -> CUfunction;
			private:
				CUDAModule* m_Module   = nullptr;
				CUfunction  m_Function = nullptr;
			};
		}
	}
}
#endif
