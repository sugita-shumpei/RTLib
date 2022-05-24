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
			class CUDANatives;
			class CUDATexture
			{
			public:
				friend class CUDANatives;
				static auto Allocate(CUDAContext* context, const CUDATextureImageCreateDesc& desc)->CUDATexture*;
				virtual ~CUDATexture()noexcept;

				void Destroy();
			public:
				CUDATexture(CUtexObject texObject)noexcept;
				auto GetCUtexObject()const noexcept -> CUtexObject;
			private:
				CUtexObject m_TexObject;
			};
		}
	}
}
#endif
