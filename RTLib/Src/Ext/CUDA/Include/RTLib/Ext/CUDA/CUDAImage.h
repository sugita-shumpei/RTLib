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
			class CUDAImage {
				friend class CUDAContext;
				friend class CUDATexture;
			public:
				static auto Allocate(CUDAContext* ctx, const CUDAImageDesc& desc)->CUDAImage*;
				virtual ~CUDAImage()noexcept;
				void Destroy()noexcept;

				auto GetImageType()const noexcept ->CUDAImageType { return m_ImageType; }
				auto GetWidth ()const noexcept -> size_t { return m_Width ; }
				auto GetHeight()const noexcept -> size_t { return m_Height; }
				auto GetDepth ()const noexcept -> size_t { return m_Depth;}
				auto GetLevels()const noexcept -> size_t { return m_Levels; }
				auto GetLayers()const noexcept -> size_t { return m_Layers; }
				auto GetFormat()const noexcept -> CUDAImageDataType { return m_Format; }
				auto GetChannels()const noexcept -> unsigned int      { return m_Channels; }
				auto GetFlags()const noexcept -> unsigned int { return m_Flags; }
				auto GetOwnership()const noexcept -> bool { return m_Ownership; }

				auto GetMipImage( unsigned int level)  -> CUDAImage*;
				auto GetMipWidth( unsigned int level)const noexcept -> size_t;
				auto GetMipHeight(unsigned int level)const noexcept -> size_t;
				auto GetMipDepth( unsigned int level)const noexcept -> size_t;
			private:
				CUDAImage(CUDAContext* ctx, const CUDAImageDesc& desc, CUarray          cuArray, bool ownership = true)noexcept;
				CUDAImage(CUDAContext* ctx, const CUDAImageDesc& desc, CUmipmappedArray cuArray, const std::vector<CUarray>& cuArrayRefs)noexcept;
				static auto AllocateArray(CUDAContext* ctx, const CUDAImageDesc& desc)->CUDAImage*;
				static auto AllocateArray3D(CUDAContext* ctx, const CUDAImageDesc& desc)->CUDAImage*;
				static auto AllocateMipmappedArray(CUDAContext* ctx, const CUDAImageDesc& desc)->CUDAImage*;
				auto GetArray()noexcept -> CUarray { return m_Arrays[0]; }
				auto GetArrays(unsigned int level)noexcept -> CUarray {
					if (level >= m_Arrays.size()) { return nullptr; }
					return m_Arrays[level]; 
				}
				auto GetMipMappedArray()noexcept -> CUmipmappedArray { return m_ArrayMipmapped; }
			private:
				CUDAContext*            m_Context;
				CUmipmappedArray        m_ArrayMipmapped;
				std::vector<CUarray>    m_Arrays;
				size_t                  m_Width;
				size_t                  m_Height;
				size_t                  m_Depth;
				size_t			        m_Levels;
				size_t                  m_Layers;
				CUDAImageDataType       m_Format;
				CUDAImageType           m_ImageType;
				unsigned int            m_Channels;
				unsigned int            m_Flags;
				bool                    m_Ownership;
			};
		}
	}
}
#endif
