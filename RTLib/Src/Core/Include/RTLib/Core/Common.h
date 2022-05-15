#ifndef RTLIB_CORE_COMMON_H
#define RTLIB_CORE_COMMON_H
#include <cstdint>
namespace RTLib {
	namespace Core {
		enum class BaseTypeFlagBits
		{
			eUndefined = 0,
			/*SizeFlag 1~32*/
			eInteger   = 1 << 6,
			eUnsigned  = 1 << 7,
			eFloat     = 1 << 8,
		};
		enum class SizedTypeFlagBits : uint64_t
		{
			eUndefined = static_cast<uint64_t>(BaseTypeFlagBits::eUndefined)  ,
			eInt8      = static_cast<uint64_t>(BaseTypeFlagBits::eInteger) |8 ,
			eInt16     = static_cast<uint64_t>(BaseTypeFlagBits::eInteger) |16,
			eInt32     = static_cast<uint64_t>(BaseTypeFlagBits::eInteger) |32,
			eUInt8     = static_cast<uint64_t>(BaseTypeFlagBits::eUnsigned)|8 ,
			eUInt16    = static_cast<uint64_t>(BaseTypeFlagBits::eUnsigned)|16,
			eUInt32    = static_cast<uint64_t>(BaseTypeFlagBits::eUnsigned)|32,
			eFloat16   = static_cast<uint64_t>(BaseTypeFlagBits::eFloat)   |16,
			eFloat32   = static_cast<uint64_t>(BaseTypeFlagBits::eFloat)   |32,
		};
		enum class AttachmentCompponent :uint64_t
		{
			eRed         = ((uint64_t)1) << 32,
			eGreen       = ((uint64_t)1) << 33,
			eBlue        = ((uint64_t)1) << 34,
			eAlpha       = ((uint64_t)1) << 35,
			eDepth       = ((uint64_t)1) << 36,
			eStencil     = ((uint64_t)1) << 37,
		};
		enum class BaseFormat :uint64_t
		{
			eBaseRed  = static_cast<uint64_t>(AttachmentCompponent::eRed),
			eBaseRG   = static_cast<uint64_t>(AttachmentCompponent::eRed)| 
						static_cast<uint64_t>(AttachmentCompponent::eGreen),
			eBaseRGB  = static_cast<uint64_t>(AttachmentCompponent::eRed)  | 
						static_cast<uint64_t>(AttachmentCompponent::eGreen)| 
						static_cast<uint64_t>(AttachmentCompponent::eBlue),
			eBaseRGBA = static_cast<uint64_t>(AttachmentCompponent::eRed)  | 
						static_cast<uint64_t>(AttachmentCompponent::eGreen)| 
						static_cast<uint64_t>(AttachmentCompponent::eBlue) | 
						static_cast<uint64_t>(AttachmentCompponent::eAlpha),

			eBaseDepth		  = static_cast<uint64_t>(AttachmentCompponent::eDepth),
			eBaseStencil      = static_cast<uint64_t>(AttachmentCompponent::eStencil),
			eBaseDepthStencil = static_cast<uint64_t>(AttachmentCompponent::eDepth)|
								static_cast<uint64_t>(AttachmentCompponent::eStencil),

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
			ImageSubresourceLayers srcImageSubresources;
			Offset3D               srcImageOffset;
			Extent3D               srcImageExtent;
		};

		struct ImageMemoryCopy
		{
			void*				   srcData;
			ImageSubresourceLayers dstImageSubresources;
			Offset3D               dstImageOffset;
			Extent3D               dstImageExtent;
		};

		struct ImageCopy
		{
			ImageSubresourceLayers srcImageSubresources;
			Offset3D               srcImageOffset;
			ImageSubresourceLayers dstImageSubresources;
			Offset3D               dstImageOffset;
			Extent3D               extent;
		};
	}
}
#endif
