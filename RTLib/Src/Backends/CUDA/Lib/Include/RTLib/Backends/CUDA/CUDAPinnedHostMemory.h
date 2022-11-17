#ifndef RTLIB_BACKENDS_CUDA_PINNED_HOST_MEMORY_H
#define RTLIB_BACKENDS_CUDA_PINNED_HOST_MEMORY_H
#include <vector>
#include <memory>
#include <any>
namespace RTLib
{
	namespace Backends {
		namespace Cuda {
			class PinnedHostMemory
			{
				friend class CurrentContext;
				PinnedHostMemory(size_t sizeInBytes) noexcept;
			public:
				~PinnedHostMemory() noexcept;

				PinnedHostMemory(PinnedHostMemory&&)noexcept = delete;
				PinnedHostMemory(const PinnedHostMemory&) = delete;
				PinnedHostMemory& operator=(PinnedHostMemory&&)noexcept = delete;
				PinnedHostMemory& operator=(const PinnedHostMemory&) = delete;

				auto GetHandle()const noexcept -> void*;
				auto GetSizeInBytes()const noexcept -> size_t;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
