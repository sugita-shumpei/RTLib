#ifndef RTLIB_BACKENDS_CUDA_CUDA_MODULE_H
#define RTLIB_BACKENDS_CUDA_CUDA_MODULE_H
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
			class Function;
			class Module {
			private:
				friend class Function;
				friend class CurrentContext;

				Module(const char* filename)noexcept;
				Module(const std::vector<char>& image)noexcept;
				Module(const std::vector<char>& image, const std::unordered_map<JitOption,void*>& options)noexcept;
			public:
				~Module() noexcept;

				Module(const Module&) = delete;
				Module& operator=(const Module&) = delete;

				Module(Module&&) = delete;
				Module& operator=(Module&&) = delete;

				auto GetHandle()const noexcept -> void*;
				auto GetFunction(const char* name) noexcept -> Function*;
				auto GetGlobal(const char* name)const noexcept -> GlobalAddressDesc;

			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};

		}
	}
}
#endif
