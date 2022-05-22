#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
#include <iostream>
#include <string>
#include <cassert>
struct RTLib::Ext::CUDA::CUDABuffer ::Impl{

	CUDAContext*    context     = nullptr;
	size_t          sizeInBytes = 0;
	CUDAMemoryFlags flags       = CUDAMemoryFlags::eDefault;
	CUdeviceptr     deviceptr   = 0;
};
auto RTLib::Ext::CUDA::CUDABuffer::Allocate(CUDAContext* ctx, const CUDABufferCreateDesc& desc) -> CUDABuffer*
{
	if (desc.sizeInBytes == 0) { return nullptr; }
	CUdeviceptr deviceptr = 0;
	if (desc.flags == CUDAMemoryFlags::eDefault) {
		RTLIB_EXT_CUDA_THROW_IF_FAILED(cuMemAlloc(&deviceptr, desc.sizeInBytes));
	}
	auto buffer = new CUDABuffer(ctx, desc,deviceptr);
	return buffer;
}

void RTLib::Ext::CUDA::CUDABuffer::Destroy() noexcept
{
	assert(m_Impl != nullptr);
	m_Impl->context = nullptr;
	m_Impl->flags = CUDAMemoryFlags::eDefault;
	m_Impl->sizeInBytes = 0;
	if (!m_Impl->deviceptr) {
		return;
	}
	try {
		RTLIB_EXT_CUDA_THROW_IF_FAILED(cuMemFree(m_Impl->deviceptr));
	}
	catch (CUDAException& err) {
		std::cerr << err.what() << std::endl;
	}
	m_Impl->deviceptr = 0;
}

RTLib::Ext::CUDA::CUDABuffer::~CUDABuffer() noexcept
{
	m_Impl.reset();
}

auto RTLib::Ext::CUDA::CUDABuffer::GetDeviceAddress() noexcept -> CUdeviceptr { return m_Impl->deviceptr; }

auto RTLib::Ext::CUDA::CUDABuffer::GetSizeInBytes() const noexcept -> size_t { return m_Impl->sizeInBytes; }

RTLib::Ext::CUDA::CUDABuffer::CUDABuffer(CUDAContext* ctx, const CUDABufferCreateDesc& desc, CUdeviceptr deviceptr) noexcept:m_Impl{new Impl()}
{
	m_Impl->context     = ctx;
	m_Impl->flags       = desc.flags;
	m_Impl->sizeInBytes = desc.sizeInBytes;
	m_Impl->deviceptr   = deviceptr;
}

