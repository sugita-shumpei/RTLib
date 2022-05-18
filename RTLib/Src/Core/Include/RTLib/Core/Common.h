#ifndef RTLIB_CORE_COMMON_H
#define RTLIB_CORE_COMMON_H
#include <RTLib/Core/TypeFormat.h>
#include <cstdint>
namespace RTLib {
	namespace Core {
		enum class FilterMode
		{
			eNearest,
			eLinear,
		};

		enum class SamplerMipmapMode
		{
			eNearest,
			eLinear,
		};

		enum class SamplerAddressMode
		{
			eRepeat,
			eMirroredRepeat,
			eClampToEdge,
			eClampToBorder,
			eMirrorClampToEdge,
		};

		enum class CompareOp
		{
			eNever = 0,
			eLess  = 1,
			eEqual = 2,
			eLessOrEqual = 3,
			eGreater  = 4,
			eNotEqual = 5,
			eGreaterOrEqual = 6,
			eAlways = 7,
		};

		struct Extent2D
		{
			uint32_t width;
			uint32_t height;
		};
		struct Extent3D
		{
			uint32_t width;
			uint32_t height;
			uint32_t depth;
		};

		struct Offset2D
		{
			int32_t x;
			int32_t y;
		};
		struct Offset3D
		{
			int32_t x;
			int32_t y;
			int32_t z;
		};

		struct ImageSubresourceLayers
		{
			uint32_t mipLevels;
			uint32_t baseArrayLayer;
			uint32_t layerCount;
		};

		struct BufferCopy
		{
			size_t srcOffset;
			size_t dstOffset;
			size_t size;
		};

		struct MemoryBufferCopy
		{
			const void* srcData;
			size_t      dstOffset;
			size_t      size;
		};

		struct BufferMemoryCopy
		{
			void*  dstData;
			size_t srcOffset;
			size_t size;
		};

		struct BufferImageCopy
		{
			size_t				   bufferOffset;
			size_t                 bufferRowLength;
			size_t                 bufferImageHeight;
			ImageSubresourceLayers imageSubresources;
			Offset3D               imageOffset;
			Extent3D               imageExtent;
		};


		struct MemoryImageCopy
		{
			const void*            srcData;
			ImageSubresourceLayers dstImageSubresources;
			Offset3D               dstImageOffset;
			Extent3D               dstImageExtent;
		};

		struct ImageMemoryCopy
		{
			void*				   dstData;
			ImageSubresourceLayers srcImageSubresources;
			Offset3D               srcImageOffset;
			Extent3D               srcImageExtent;
		};

		struct ImageCopy
		{
			ImageSubresourceLayers srcImageSubresources;
			Offset3D               srcImageOffset;
			ImageSubresourceLayers dstImageSubresources;
			Offset3D               dstImageOffset;
			Extent3D               extent;
		};

		struct SamplerCreateDesc
		{
			FilterMode              magFilter;
			FilterMode              minFilter;
			SamplerMipmapMode       mipmapMode;
		    SamplerAddressMode      addressModeU;
			SamplerAddressMode      addressModeV;
			SamplerAddressMode      addressModeW;
			float                   mipLodBias;
			bool                    anisotropyEnable;
			float                   maxAnisotropy;
			bool                    compareEnable;
			CompareOp               compareOp;
			float                   minLod;
			float                   maxLod;
			float                   borderColor[4];
			bool                    unnormalizedCoordinates;
		};
	}
}
#endif
