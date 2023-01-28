#ifndef RTLIB_EXT_CUDA_CUDA_BUFFER_H
#define RTLIB_EXT_CUDA_CUDA_BUFFER_H
#include <RTLib/Core/Buffer.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <RTLib/Ext/CUDA/UuidDefinitions.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
namespace RTLib {
	namespace Ext
	{
		namespace CUDA
		{
			class CUDAContext;
			class CUDABufferView;
			class CUDABuffer;
			class CUDANatives;
			class CUDABuffer : public Core::Buffer{
				friend class CUDABufferView;
				friend class CUDANatives;
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(CUDABuffer, Core::Buffer, RTLIB_TYPE_UUID_RTLIB_EXT_CUDA_CUDA_BUFFER);
				static auto Allocate(CUDAContext* ctx, const CUDABufferCreateDesc& desc)->CUDABuffer*;
				virtual void Destroy()noexcept override;

				virtual ~CUDABuffer()noexcept;
				auto GetSizeInBytes() const noexcept -> size_t;
			private:
				CUDABuffer(CUDAContext* ctx, const CUDABufferCreateDesc& desc, CUdeviceptr deviceptr, bool hasOwnership = true) noexcept;
				auto GetCUdeviceptr() const noexcept -> CUdeviceptr;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			struct CUDABufferView
			{
			public:
				friend class CUDANatives;
			public:
				CUDABufferView()noexcept :m_Base{ nullptr }, m_OffsetInBytes{ 0 }, m_SizeInBytes{ 0 }{}
				CUDABufferView(CUDABuffer* base)noexcept;
				CUDABufferView(CUDABuffer* base, size_t offsetInBytes, size_t sizeInBytes)noexcept;
				CUDABufferView(const CUDABufferView& bufferView, size_t offsetInBytes, size_t sizeInBytes)noexcept;

				CUDABufferView(const CUDABufferView& bufferView)noexcept;
				CUDABufferView& operator=(const CUDABufferView& bufferView)noexcept;

				auto GetBaseBuffer()const noexcept -> const CUDABuffer* { return m_Base; }
				auto GetBaseBuffer()      noexcept ->       CUDABuffer* { return m_Base; }
				auto GetSizeInBytes() const noexcept -> size_t { return m_SizeInBytes; }
				auto GetOffsetInBytes()const noexcept-> size_t { return m_OffsetInBytes; }
			private:
				auto GetCUdeviceptr()const noexcept -> CUdeviceptr;
			private:
				CUDABuffer* m_Base;
				size_t      m_OffsetInBytes;
				size_t      m_SizeInBytes;
			};
		}
	}
}
#endif
