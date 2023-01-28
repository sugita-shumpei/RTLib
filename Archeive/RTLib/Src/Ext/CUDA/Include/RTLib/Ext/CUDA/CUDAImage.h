#ifndef RTLIB_EXT_CUDA_CUDA_IMAGE_H
#define RTLIB_EXT_CUDA_CUDA_IMAGE_H
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <vector>
namespace RTLib {
	namespace Ext
	{
		namespace CUDA
		{
			class CUDAContext;
			class CUDANatives;
			class CUDAImage {
				friend class CUDAContext;
				friend class CUDAStream ;
				friend class CUDATexture;
				friend class CUDANatives;
			public:
				static auto Allocate(CUDAContext* ctx, const CUDAImageCreateDesc& desc)->CUDAImage*;
				virtual ~CUDAImage()noexcept;
				void Destroy()noexcept;

				auto GetImageType()const noexcept ->CUDAImageType;
				auto GetWidth()const noexcept -> size_t;
				auto GetHeight()const noexcept -> size_t;
				auto GetDepth()const noexcept -> size_t;
				auto GetLevels()const noexcept -> size_t;
				auto GetLayers()const noexcept -> size_t;
				auto GetFormat()const noexcept -> CUDAImageFormat;
				auto GetFlags()const noexcept -> unsigned int;
				auto GetOwnership()const noexcept -> bool;

				auto GetMipImage( unsigned int level)  -> CUDAImage*;
				auto GetMipWidth( unsigned int level)const noexcept -> size_t;
				auto GetMipHeight(unsigned int level)const noexcept -> size_t;
				auto GetMipDepth( unsigned int level)const noexcept -> size_t;
			private:
				CUDAImage(CUDAContext* ctx, const CUDAImageCreateDesc& desc, CUarray          cuArray, bool ownership = true)noexcept;
				CUDAImage(CUDAContext* ctx, const CUDAImageCreateDesc& desc, CUmipmappedArray cuArray, const std::vector<CUarray>& cuArrayRefs, bool ownership = true)noexcept;
				static auto AllocateArray(CUDAContext* ctx, const CUDAImageCreateDesc& desc)->CUDAImage*;
				static auto AllocateArray3D(CUDAContext* ctx, const CUDAImageCreateDesc& desc)->CUDAImage*;
				static auto AllocateMipmappedArray(CUDAContext* ctx, const CUDAImageCreateDesc& desc)->CUDAImage*;
				auto GetCUarray()noexcept -> CUarray;
				auto GetCUarrayWithLevel(unsigned int level)noexcept -> CUarray;
				auto GetCUmipmappedArray()noexcept -> CUmipmappedArray;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
