#include <RTLib/Backends/CUDA/CUDALinearMemory.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include "CUDAInternals.h"
struct RTLib::Backends::Cuda::LinearMemory1D::Impl {
	Impl(size_t sizeInBytes_) noexcept:sizeInBytes{ sizeInBytes_ }
	{
		assert(CurrentContext::Handle().Get());
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemAlloc(&ptr, sizeInBytes_));
	}
	~Impl()noexcept {
		assert(CurrentContext::Handle().Get());
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemFree(ptr));
	}
	CUdeviceptr ptr = 0;
	size_t sizeInBytes = 0;
};

RTLib::Backends::Cuda::LinearMemory1D::LinearMemory1D(size_t sizeInBytes) noexcept:LinearMemory(),m_Impl{new Impl(sizeInBytes)}
{
}

RTLib::Backends::Cuda::LinearMemory1D::~LinearMemory1D() noexcept
{
	m_Impl.reset();
}

auto RTLib::Backends::Cuda::LinearMemory1D::GetHandle() const noexcept -> void*
{
	return reinterpret_cast<void*>(m_Impl->ptr);
}

auto RTLib::Backends::Cuda::LinearMemory1D::GetSizeInBytes() const noexcept -> size_t
{
	return size_t(m_Impl->sizeInBytes);
}

struct RTLib::Backends::Cuda::LinearMemory2D::Impl {
	Impl(size_t width_, size_t height_, unsigned int elementSizeInBytes_) noexcept 
		:width{ width_ }, height{ height_ }, pitchSizeInBytes{ 0 }, elementSizeInBytes{ elementSizeInBytes_ }
	{
		assert(CurrentContext::Handle().Get());
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemAllocPitch(&ptr, &pitchSizeInBytes, width_ * elementSizeInBytes_, height_, elementSizeInBytes_));
	}
	~Impl()noexcept {
		assert(CurrentContext::Handle().Get());
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemFree(ptr));
	}
	CUdeviceptr ptr = 0;
	size_t width  = 0;
	size_t height = 0;
	size_t pitchSizeInBytes = 0;
	unsigned int elementSizeInBytes = 0;
};

RTLib::Backends::Cuda::LinearMemory2D::LinearMemory2D(size_t width, size_t height, unsigned int elementSizeInBytes) noexcept
	:LinearMemory(),m_Impl{new Impl(width,height,elementSizeInBytes)}
{
}

RTLib::Backends::Cuda::LinearMemory2D::~LinearMemory2D() noexcept
{
	m_Impl.reset();
}

auto RTLib::Backends::Cuda::LinearMemory2D::GetHandle() const noexcept -> void*
{
	return reinterpret_cast<void*>(m_Impl->ptr);
}

auto RTLib::Backends::Cuda::LinearMemory2D::GetSizeInBytes() const noexcept -> size_t 
{
	return GetPitchSizeInBytes() * GetHeight();
}

auto RTLib::Backends::Cuda::LinearMemory2D::GetWidth() const noexcept -> size_t
{
	return size_t(m_Impl->width);
}

auto RTLib::Backends::Cuda::LinearMemory2D::GetHeight() const noexcept -> size_t
{
	return size_t(m_Impl->height);
}

auto RTLib::Backends::Cuda::LinearMemory2D::GetElementSizeInBytes() const noexcept -> unsigned int
{
	return m_Impl->elementSizeInBytes;
}

auto RTLib::Backends::Cuda::LinearMemory2D::GetPitchSizeInBytes() const noexcept -> size_t
{
	return size_t(m_Impl->pitchSizeInBytes);
}
