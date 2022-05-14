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
			public:
				static auto Allocate(CUDAContext* ctx, const CUDABufferDesc& desc)->CUDABuffer*;
				void Destroy()noexcept;

				virtual ~CUDABuffer()noexcept;
			private:
				CUDABuffer(CUDAContext* ctx, const CUDABufferDesc& desc, CUdeviceptr deviceptr, void* hostptr) noexcept;
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
