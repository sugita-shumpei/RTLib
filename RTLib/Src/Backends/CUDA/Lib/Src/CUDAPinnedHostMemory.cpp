#include <RTLib/Backends/CUDA/CUDAPinnedHostMemory.h>
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include "CUDAInternals.h"
struct RTLib::Backends::Cuda::PinnedHostMemory::Impl {
	Impl(size_t sizeInBytes_) noexcept :sizeInBytes{ sizeInBytes_ }
	{
		assert(CurrentContext::Handle().Get());
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemAllocHost(&ptr, sizeInBytes_));
	}
	~Impl()noexcept {
		assert(CurrentContext::Handle().Get());
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemFreeHost(ptr));
	}
	void* ptr = 0;
	size_t sizeInBytes = 0;
};
RTLib::Backends::Cuda::PinnedHostMemory::PinnedHostMemory(size_t sizeInBytes) noexcept:m_Impl{new Impl(sizeInBytes)}
{
}

RTLib::Backends::Cuda::PinnedHostMemory::~PinnedHostMemory() noexcept
{
	m_Impl.reset();
}

auto RTLib::Backends::Cuda::PinnedHostMemory::GetHandle() const noexcept -> void*
{
	return m_Impl->ptr;
}

auto RTLib::Backends::Cuda::PinnedHostMemory::GetSizeInBytes() const noexcept -> size_t
{
	return m_Impl->sizeInBytes;
}
