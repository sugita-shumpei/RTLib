#ifndef RTLIB_EXT_CUDA_CUDA_BUFFER_H
#define RTLIB_EXT_CUDA_CUDA_BUFFER_H
#include <RTLib/Core/Buffer.h>
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <RTLib/Ext/CUDA/UuidDefinitions.h>
namespace RTLib {
	namespace Ext
	{
		namespace CUDA
		{
			class CUDAContext;
			class CUDABuffer : public Core::Buffer{
				friend class CUDAContext;
				friend class CUDAStream ;
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(CUDABuffer, Core::Buffer, RTLIB_TYPE_UUID_RTLIB_EXT_CUDA_CUDA_BUFFER);
				static auto Allocate(CUDAContext* ctx, const CUDABufferCreateDesc& desc)->CUDABuffer*;
				virtual void Destroy()noexcept override;

				virtual ~CUDABuffer()noexcept;
				auto GetDeviceAddress() noexcept -> CUdeviceptr;
				auto GetSizeInBytes()const noexcept -> size_t;
			private:
				CUDABuffer(CUDAContext* ctx, const CUDABufferCreateDesc& desc, CUdeviceptr deviceptr) noexcept;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			struct CUDABufferView
			{
			public:
				CUDABufferView()noexcept :m_Base{ nullptr }, m_OffsetInBytes{ 0 }, m_SizeInBytes{ 0 }{}
				CUDABufferView(CUDABuffer* base)noexcept
				{
					m_Base          = base;
					m_OffsetInBytes = 0;
					m_SizeInBytes   = base->GetSizeInBytes();
				}
				CUDABufferView(CUDABuffer* base, size_t offsetInBytes, size_t sizeInBytes)noexcept
				{
					auto realBufferSize = base->GetSizeInBytes();
					m_Base = base;
					m_OffsetInBytes = std::min(offsetInBytes, realBufferSize);
					m_SizeInBytes = std::max(std::min(sizeInBytes + m_OffsetInBytes, realBufferSize), m_OffsetInBytes) - m_OffsetInBytes;
				}
				CUDABufferView(const CUDABufferView& bufferView, size_t offsetInBytes, size_t sizeInBytes)noexcept
					:CUDABufferView(bufferView.m_Base, bufferView.m_OffsetInBytes + offsetInBytes, bufferView.m_SizeInBytes + sizeInBytes) {}

				CUDABufferView(const CUDABufferView& bufferView)noexcept
				{
					m_Base = bufferView.m_Base;
					m_OffsetInBytes = bufferView.m_OffsetInBytes;
					m_SizeInBytes = bufferView.m_SizeInBytes;
				}
				CUDABufferView& operator=(const CUDABufferView& bufferView)noexcept
				{
					if (this != &bufferView) {
						m_Base = bufferView.m_Base;
						m_OffsetInBytes = bufferView.m_OffsetInBytes;
						m_SizeInBytes = bufferView.m_SizeInBytes;
					}
					return *this;
				}

				auto GetBaseBuffer()const noexcept -> const CUDABuffer* { return m_Base; }
				auto GetBaseBuffer()      noexcept ->       CUDABuffer* { return m_Base; }
				auto GetDeviceAddress()noexcept -> CUdeviceptr { return m_Base->GetDeviceAddress() + m_OffsetInBytes; }
				auto GetSizeInBytes() const noexcept -> size_t { return m_SizeInBytes; }
				auto GetOffsetInBytes()const noexcept-> size_t { return m_OffsetInBytes; }
			private:
				CUDABuffer* m_Base;
				size_t      m_OffsetInBytes;
				size_t      m_SizeInBytes;
			};
		}
	}
}
#endif
