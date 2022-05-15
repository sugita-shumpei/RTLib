#ifndef RTLIB_EXT_CUDA_CUDA_BUFFER_H
#define RTLIB_EXT_CUDA_CUDA_BUFFER_H
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
namespace RTLib {
	namespace Ext
	{
		namespace CUDA
		{
			class CUDAContext;
			class CUDABuffer {
				friend class CUDAContext;
			public:
				static auto Allocate(CUDAContext* ctx, const CUDABufferDesc& desc)->CUDABuffer*;
				void Destroy()noexcept;

				virtual ~CUDABuffer()noexcept;
			private:
				CUDABuffer(CUDAContext* ctx, const CUDABufferDesc& desc, CUdeviceptr deviceptr, void* hostptr) noexcept;
				auto GetSizeInBytes()const noexcept -> size_t { return m_SizeInBytes; }
			private:
				auto GetDeviceAddress() noexcept -> CUdeviceptr { return m_Deviceptr; }
				auto GetHostptr() noexcept -> void* { return m_Hostptr; }
			private:
				CUDAContext*    m_Context;
				size_t          m_SizeInBytes;
				CUDAMemoryFlags m_flags;
				CUdeviceptr     m_Deviceptr;
				void*			m_Hostptr;
			};
		}
	}
}
#endif
