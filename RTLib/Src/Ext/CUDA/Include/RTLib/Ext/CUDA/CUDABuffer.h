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
		}
	}
}
#endif
