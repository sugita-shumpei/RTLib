#ifndef RTLIB_BACKENDS_CUDA_CUDA_FUNCTION_H
#define RTLIB_BACKENDS_CUDA_CUDA_FUNCTION_H
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <any>
namespace RTLib
{
	namespace Backends
	{
		namespace Cuda
		{
			class Function
			{
				friend class Module;
				Function(void* pHandle) noexcept;
			public:
				~Function() noexcept;

				Function(const Function&) = delete;
				Function& operator=(const Function&) = delete;

				Function(Module&&) = delete;
				Function& operator=(Function&&) = delete;

				auto GetHandle()const noexcept -> void*;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			
		}
	}
}
#endif
