#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
#include <iostream>
#include <string>
#include <cassert>
struct RTLib::Ext::CUDA::CUDABuffer ::Impl{

	CUDAContext*    context     = nullptr;
	size_t          sizeInBytes = 0;
	CUDAMemoryFlags flags       = CUDAMemoryFlags::eDefault;
	CUdeviceptr     deviceptr   = 0;
	bool            hasOwnership= true;
};
auto RTLib::Ext::CUDA::CUDABuffer::Allocate(CUDAContext* ctx, const CUDABufferCreateDesc& desc) -> CUDABuffer*
{
	if (desc.sizeInBytes == 0) { return nullptr; }
	CUdeviceptr deviceptr = 0;
	if (desc.flags == CUDAMemoryFlags::eDefault) {
		RTLIB_EXT_CUDA_THROW_IF_FAILED(cuMemAlloc(&deviceptr, desc.sizeInBytes));
	}
	if (desc.pData) {
		RTLIB_EXT_CUDA_THROW_IF_FAILED(cuMemcpyHtoD(deviceptr, desc.pData, desc.sizeInBytes));
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
	if (m_Impl->hasOwnership) {
		try {
			RTLIB_EXT_CUDA_THROW_IF_FAILED(cuMemFree(m_Impl->deviceptr));
		}
		catch (CUDAException& err) {
			std::cerr << err.what() << std::endl;
		}
	}
	m_Impl->deviceptr = 0;
}

RTLib::Ext::CUDA::CUDABuffer::~CUDABuffer() noexcept
{
	m_Impl.reset();
}

auto RTLib::Ext::CUDA::CUDABuffer::GetSizeInBytes() const noexcept -> size_t { return m_Impl->sizeInBytes; }

RTLib::Ext::CUDA::CUDABuffer::CUDABuffer(CUDAContext* ctx, const CUDABufferCreateDesc& desc, CUdeviceptr deviceptr, bool hasOwnership) noexcept:m_Impl{new Impl()}
{
	m_Impl->context     = ctx;
	m_Impl->flags       = desc.flags;
	m_Impl->sizeInBytes = desc.sizeInBytes;
	m_Impl->deviceptr   = deviceptr;
	m_Impl->hasOwnership= hasOwnership;
}

auto RTLib::Ext::CUDA::CUDABuffer::GetCUdeviceptr() const noexcept -> CUdeviceptr { return m_Impl->deviceptr; }

RTLib::Ext::CUDA::CUDABufferView::CUDABufferView(CUDABuffer* base) noexcept
{
	m_Base = base;
	m_OffsetInBytes = 0;
	m_SizeInBytes = base->GetSizeInBytes();
}

RTLib::Ext::CUDA::CUDABufferView::CUDABufferView(CUDABuffer* base, size_t offsetInBytes, size_t sizeInBytes) noexcept
{
	auto realBufferSize = base->GetSizeInBytes();
	m_Base = base;
	m_OffsetInBytes = std::min(offsetInBytes, realBufferSize);
	m_SizeInBytes = std::max(std::min(sizeInBytes + m_OffsetInBytes, realBufferSize), m_OffsetInBytes) - m_OffsetInBytes;
}

RTLib::Ext::CUDA::CUDABufferView::CUDABufferView(const CUDABufferView& bufferView, size_t offsetInBytes, size_t sizeInBytes) noexcept
	:CUDABufferView(bufferView.m_Base, bufferView.m_OffsetInBytes + offsetInBytes, bufferView.m_SizeInBytes + sizeInBytes) {}

RTLib::Ext::CUDA::CUDABufferView::CUDABufferView(const CUDABufferView& bufferView) noexcept
{
	m_Base = bufferView.m_Base;
	m_OffsetInBytes = bufferView.m_OffsetInBytes;
	m_SizeInBytes = bufferView.m_SizeInBytes;
}

auto RTLib::Ext::CUDA::CUDABufferView::operator=(const CUDABufferView& bufferView) noexcept->CUDABufferView&
{
	if (this != &bufferView) {
		m_Base = bufferView.m_Base;
		m_OffsetInBytes = bufferView.m_OffsetInBytes;
		m_SizeInBytes = bufferView.m_SizeInBytes;
	}
	return *this;
}

auto RTLib::Ext::CUDA::CUDABufferView::GetCUdeviceptr() const noexcept -> CUdeviceptr { return m_Base->GetCUdeviceptr() + m_OffsetInBytes; }
