#ifndef RTLIB_EXT_CUDA_CUDA_TEXTURE_H
#define RTLIB_EXT_CUDA_CUDA_TEXTURE_H
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
namespace RTLib
{
	namespace Ext
	{
		namespace CUDA
		{
			class CUDATexture
			{
			public:
				static auto Allocate(CUDAContext* context, const CUDATextureImageDesc& desc)->CUDATexture*;
				virtual ~CUDATexture()noexcept;

				void Destroy();
			public:
				CUDATexture(CUtexObject texObject)noexcept;
			private:
				CUtexObject m_TexObject;
			};
		}
	}
}
#endif
