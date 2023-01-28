#ifndef RTLIB_EXT_CUDA_CUDA_MODULE_H
#define RTLIB_EXT_CUDA_CUDA_MODULE_H
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <vector>
namespace RTLib
{
	namespace Ext
	{
		namespace CUDA
		{
			class  CUDAContext;
			class  CUDAFunction;
			struct CUDANatives;
			class  CUDAModule
			{
				friend class CUDANatives;
			public:
				static auto LoadFromFile(CUDAContext* context, const char* filename)->CUDAModule*;
				static auto LoadFromData(CUDAContext* context, const void* data)->CUDAModule*;
				static auto LoadFromData(CUDAContext* context, const void* data, const std::vector<CUDAJitOptionValue>& optionValues)->CUDAModule*;
				virtual ~CUDAModule()noexcept;

				void Destory()noexcept;

				auto LoadFunction(const char* entryPoint)->CUDAFunction*;
			private:
				CUDAModule(CUDAContext* context, CUmodule cuModule)noexcept;
				auto GetCUmodule()noexcept -> CUmodule;
			private:
				CUDAContext* m_Context = nullptr;
				CUmodule     m_Module  = nullptr;
			};
		}
	}
}
#endif
