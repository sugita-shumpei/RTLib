#ifndef RTLIB_CORE_COMMON_H
#define RTLIB_CORE_COMMON_H
#include <cstdint>
#define RTLIB_CORE_CORE_FORMAT_DEF_1(VAL1) ((uint64_t)VAL1)
#define RTLIB_CORE_CORE_FORMAT_DEF_2(VAL1, VAL2) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9)
#define RTLIB_CORE_CORE_FORMAT_DEF_3(VAL1, VAL2, VAL3) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18)
#define RTLIB_CORE_CORE_FORMAT_DEF_4(VAL1, VAL2, VAL3, VAL4) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18) | (((uint64_t)VAL4) << 27)
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
			eInt8X1    = RTLIB_CORE_CORE_FORMAT_DEF_1(eInt8),
			eInt8X2    = RTLIB_CORE_CORE_FORMAT_DEF_2(eInt8,eInt8),
			eInt8X3    = RTLIB_CORE_CORE_FORMAT_DEF_3(eInt8,eInt8,eInt8),
			eInt8X4    = RTLIB_CORE_CORE_FORMAT_DEF_4(eInt8,eInt8,eInt8,eInt8),
			eUInt8X1   = RTLIB_CORE_CORE_FORMAT_DEF_1(eUInt8),
			eUInt8X2   = RTLIB_CORE_CORE_FORMAT_DEF_2(eUInt8, eUInt8),
			eUInt8X3   = RTLIB_CORE_CORE_FORMAT_DEF_3(eUInt8, eUInt8, eUInt8),
			eUInt8X4   = RTLIB_CORE_CORE_FORMAT_DEF_4(eUInt8, eUInt8, eUInt8, eUInt8),
			eInt16X1   = RTLIB_CORE_CORE_FORMAT_DEF_1(eInt16),
			eInt16X2   = RTLIB_CORE_CORE_FORMAT_DEF_2(eInt16,eInt16),
			eInt16X3   = RTLIB_CORE_CORE_FORMAT_DEF_3(eInt16,eInt16,eInt16),
			eInt16X4   = RTLIB_CORE_CORE_FORMAT_DEF_4(eInt16,eInt16,eInt16,eInt16),
			eUInt16X1  = RTLIB_CORE_CORE_FORMAT_DEF_1(eUInt16),
			eUInt16X2  = RTLIB_CORE_CORE_FORMAT_DEF_2(eUInt16, eUInt16),
			eUInt16X3  = RTLIB_CORE_CORE_FORMAT_DEF_3(eUInt16, eUInt16, eUInt16),
			eUInt16X4  = RTLIB_CORE_CORE_FORMAT_DEF_4(eUInt16, eUInt16, eUInt16, eUInt16),
			eInt32X1   = RTLIB_CORE_CORE_FORMAT_DEF_1(eInt32),
			eInt32X2   = RTLIB_CORE_CORE_FORMAT_DEF_2(eInt32,eInt32),
			eInt32X3   = RTLIB_CORE_CORE_FORMAT_DEF_3(eInt32,eInt32,eInt32),
			eInt32X4   = RTLIB_CORE_CORE_FORMAT_DEF_4(eInt32,eInt32,eInt32,eInt32),
			eUInt32X1  = RTLIB_CORE_CORE_FORMAT_DEF_1(eUInt32),
			eUInt32X2  = RTLIB_CORE_CORE_FORMAT_DEF_2(eUInt32, eUInt32),
			eUInt32X3  = RTLIB_CORE_CORE_FORMAT_DEF_3(eUInt32, eUInt32, eUInt32),
			eUInt32X4  = RTLIB_CORE_CORE_FORMAT_DEF_4(eUInt32, eUInt32, eUInt32, eUInt32),
			eFloat16X1 = RTLIB_CORE_CORE_FORMAT_DEF_1(eFloat16),
			eFloat16X2 = RTLIB_CORE_CORE_FORMAT_DEF_2(eFloat16, eFloat16),
			eFloat16X3 = RTLIB_CORE_CORE_FORMAT_DEF_3(eFloat16, eFloat16, eFloat16),
			eFloat16X4 = RTLIB_CORE_CORE_FORMAT_DEF_4(eFloat16, eFloat16, eFloat16, eFloat16),
			eFloat32X1 = RTLIB_CORE_CORE_FORMAT_DEF_1(eFloat32),
			eFloat32X2 = RTLIB_CORE_CORE_FORMAT_DEF_2(eFloat32, eFloat32),
			eFloat32X3 = RTLIB_CORE_CORE_FORMAT_DEF_3(eFloat32, eFloat32, eFloat32),
			eFloat32X4 = RTLIB_CORE_CORE_FORMAT_DEF_4(eFloat32, eFloat32, eFloat32, eFloat32),
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
#undef RTLIB_CORE_CORE_FORMAT_DEF_1
#undef RTLIB_CORE_CORE_FORMAT_DEF_2
#undef RTLIB_CORE_CORE_FORMAT_DEF_3
#undef RTLIB_CORE_CORE_FORMAT_DEF_4
#endif
