#ifndef RTLIB_EXT_CUDA_CUDA_IMAGE_H
#define RTLIB_EXT_CUDA_CUDA_IMAGE_H
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
namespace RTLib {
	namespace Ext
	{
		namespace CUDA
		{
			class CUDAContext;
			class CUDAImage {
			public:
				static auto Allocate(CUDAContext* ctx, const CUDAImageDesc& desc)->CUDAImage*;
				virtual ~CUDAImage()noexcept;
			private:
				CUarray	     	m_Array;
				size_t          m_Width;
				size_t          m_Height;
				size_t          m_Depth;
				size_t          m_Layers;
				CUDAImageDataType m_Format;
				unsigned int    m_NumChannels;
				unsigned int    m_Flags;
			};
		}
	}
}
#endif
