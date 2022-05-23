#ifndef RTLIB_CORE_COMMON_H
#define RTLIB_CORE_COMMON_H
#include <RTLib/Core/TypeFormat.h>
#include <cstdint>
#include <cstddef>
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
			uint32_t mipLevel;
			uint32_t baseArrayLayer;
			uint32_t layerCount;
		};
		struct ImageSubresourceRange {
			uint32_t baseMipLevels;
			uint32_t levelCount;
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
			FilterMode              magFilter        = FilterMode::eLinear;
			FilterMode              minFilter        = FilterMode::eNearest;
			SamplerMipmapMode       mipmapMode       = SamplerMipmapMode::eLinear;
			SamplerAddressMode      addressModeU     = SamplerAddressMode::eRepeat;
			SamplerAddressMode      addressModeV     = SamplerAddressMode::eRepeat;
			SamplerAddressMode      addressModeW     = SamplerAddressMode::eRepeat;
			float                   mipLodBias       = 0.0f;
			bool                    anisotropyEnable = false;
			float                   maxAnisotropy    = 1.0f;
			bool                    compareEnable    = false;
			CompareOp               compareOp        = CompareOp::eNever;
			float                   minLod           = -1000.0f;
			float                   maxLod           =  1000.0f;
			float                   borderColor[4]   = {};
			bool                    unnormalizedCoordinates = false;
		};

		enum ShaderStageFlagBits {
			ShaderStageVertex          = 1 << 0,
			ShaderStageGeometry        = 1 << 1,
			ShaderStageTessControl     = 1 << 2,
			ShaderStageTessEvalauation = 1 << 3,
			ShaderStageFragment        = 1 << 4,
			ShaderStageCompute         = 1 << 5,
			ShaderStageRayGen          = 1 << 6,
			ShaderStageMiss            = 1 << 7,
			ShaderStageClosesthit      = 1 << 8,
			ShaderStageAnyhit          = 1 << 9,
			ShaderStageIntersection    = 1 <<10,
			ShaderStageMesh            = 1 <<11,
		};

		using ShaderStageFlags = unsigned int;
	}
}
#endif
