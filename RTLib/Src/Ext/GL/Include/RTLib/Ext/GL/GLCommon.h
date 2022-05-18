#ifndef RTLIB_EXT_GL_GL_COMMON_H
#define RTLIB_EXT_GL_GL_COMMON_H
#include <RTLib/Core/Common.h>
#include <RTLib/Ext/GL/GLTypeFormat.h>
#include <half.h>
#include <glad/glad.h>
#include <cstdint>

namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			enum class GLBindingState
			{
				eBinded,
				eUndefined,
				eUnbinded
			};

			using GLExtent2D = Core::Extent2D;
			using GLExtent3D = Core::Extent3D;

			using GLOffset2D = Core::Offset2D;
			using GLOffset3D = Core::Offset3D;

			using GLImageSubresourceLayers = Core::ImageSubresourceLayers;
			using GLBufferCopy       = Core::BufferCopy;
			using GLBufferImageCopy  = Core::BufferImageCopy;
			using GLMemoryBufferCopy = Core::MemoryBufferCopy;
			using GLBufferMemoryCopy = Core::BufferMemoryCopy;
			using GLImageMemoryCopy  = Core::ImageMemoryCopy;

			enum  GLBufferUsageFlagBits : unsigned int
			{
				GLBufferUsageUnknown           = 0,
				GLBufferUsageVertex            = 1 << 0,
				GLBufferUsageAtomicCounter     = 1 << 1,
				GLBufferUsageDispatchIndirect  = 1 << 2,
				GLBufferUsageDrawIndirect      = 1 << 3,
				GLBufferUsageIndex             = 1 << 4,
				GLBufferUsageQuery             = 1 << 5,
				GLBufferUsageStorage           = 1 << 6,
				GLBufferUsageTexture           = 1 << 7,
				GLBufferUsageTransformFeedBack = 1 << 8,
				GLBufferUsageUniform           = 1 << 9,
				GLBufferUsageImageCopySrc	   = 1 << 10,
				GLBufferUsageImageCopyDst      = 1 << 11,
			}; 
			using GLBufferUsageFlags = unsigned int;

			inline constexpr auto GetGLBufferUsageCount(GLBufferUsageFlags  usageFlags)->unsigned int {
				unsigned int count = 0;
				if ((usageFlags & GLBufferUsageVertex)            == GLBufferUsageVertex)            { ++count; }
				if ((usageFlags & GLBufferUsageIndex)             == GLBufferUsageIndex)             { ++count; }
				if ((usageFlags & GLBufferUsageUniform)           == GLBufferUsageUniform)           { ++count; }
				if ((usageFlags & GLBufferUsageStorage)           == GLBufferUsageStorage)           { ++count; }
				if ((usageFlags & GLBufferUsageImageCopySrc)      == GLBufferUsageImageCopySrc)      { ++count; }
				if ((usageFlags & GLBufferUsageImageCopyDst)      == GLBufferUsageImageCopyDst)      { ++count; }
				if ((usageFlags & GLBufferUsageDrawIndirect)      == GLBufferUsageDrawIndirect)      { ++count; }
				if ((usageFlags & GLBufferUsageDispatchIndirect)  == GLBufferUsageDispatchIndirect)  { ++count; }
				if ((usageFlags & GLBufferUsageTransformFeedBack) == GLBufferUsageTransformFeedBack) { ++count; }
				if ((usageFlags & GLBufferUsageTexture)           == GLBufferUsageTexture)           { ++count; }
				if ((usageFlags & GLBufferUsageQuery)             == GLBufferUsageQuery)             { ++count; }
				if ((usageFlags & GLBufferUsageAtomicCounter)     == GLBufferUsageAtomicCounter)     { ++count; }
				return count;
			}
			inline constexpr auto GetGLBufferMainUsage (GLBufferUsageFlags  usageFlags)->GLBufferUsageFlagBits
			{
				if ((usageFlags & GLBufferUsageVertex)	          == GLBufferUsageVertex) {
					return GLBufferUsageVertex;
				}
				if ((usageFlags & GLBufferUsageIndex)             == GLBufferUsageIndex) {
					return GLBufferUsageIndex;
				}
				if ((usageFlags & GLBufferUsageUniform)           == GLBufferUsageUniform) {
					return GLBufferUsageUniform;
				}
				if ((usageFlags & GLBufferUsageStorage)           == GLBufferUsageStorage) {
					return GLBufferUsageStorage;
				}
				if ((usageFlags & GLBufferUsageImageCopySrc)      == GLBufferUsageImageCopySrc) {
					return GLBufferUsageImageCopySrc;
				}
				if ((usageFlags & GLBufferUsageImageCopyDst)      == GLBufferUsageImageCopyDst) {
					return GLBufferUsageImageCopyDst;
				}
				if ((usageFlags & GLBufferUsageDrawIndirect)      == GLBufferUsageDrawIndirect) {
					return GLBufferUsageDrawIndirect;
				}
				if ((usageFlags & GLBufferUsageDispatchIndirect)  == GLBufferUsageDispatchIndirect) {
					return GLBufferUsageDispatchIndirect;
				}
				if ((usageFlags & GLBufferUsageTransformFeedBack) == GLBufferUsageTransformFeedBack) {
					return GLBufferUsageTransformFeedBack;
				}
				if ((usageFlags & GLBufferUsageTexture)           == GLBufferUsageTexture) {
					return GLBufferUsageTexture;
				}
				if ((usageFlags & GLBufferUsageQuery)             == GLBufferUsageQuery) {
					return GLBufferUsageQuery;
				}
				if ((usageFlags & GLBufferUsageAtomicCounter)     == GLBufferUsageAtomicCounter) {
					return GLBufferUsageAtomicCounter;
				}
				return GLBufferUsageUnknown;
			}
			enum  GLMemoryPropertyFlagBits :unsigned int
			{
				GLMemoryPropertyDeviceLocal = 1<<0,
				GLMemoryPropertyHostVisible = 1<<1,
				GLMemoryPropertyHostCoherent= 1<<2,
				GLMemoryPropertyHostCache   = 1<<3,
			};
			using GLMemoryPropertyFlags = unsigned int;

			struct GLBufferCreateDesc
			{
				size_t                 size;
				GLBufferUsageFlags    usage;
				GLMemoryPropertyFlags props;
				const void*           pData;
			};

			enum class GLImageType
			{
				e1D,
				e2D,
				e3D
			};
			///                    1
			//                     Snorm
			namespace Test {
				struct GLTypeTestCase
				{
					static inline constexpr auto v1 = GetGLTypeBaseType(GLTypeFlagBits::eFloat32);
					static_assert(v1 == GLTypeFlagBits::eFloat32);
					static inline constexpr auto v2 = GetGLTypeBaseType(GLTypeFlagBits::eUInt16_4_4_4_4);
					static_assert(v2 == GLTypeFlagBits::eUInt16);
					static inline constexpr auto v3 = GetGLTypeBaseType(GLTypeFlagBits::eUInt32_2_10_10_10_Rev);
					static_assert(v3 == GLTypeFlagBits::eUInt32);
					static inline constexpr auto v4 = GetGLTypeChannelSize(GLTypeFlagBits::eUInt32_2_10_10_10_Rev, 0);
					static_assert(v4 == 2);
					static inline constexpr auto v5 = GetGLTypeChannelSize(GLTypeFlagBits::eUInt32_2_10_10_10_Rev, 1);
					static_assert(v5 == 10);
				};
			}

			namespace Test {
				struct GLFormatTestCase
				{
					static inline constexpr auto v1 = GetGLFormatBaseFormat(GLFormat::eRG8UI);
					static inline constexpr auto v2 = GetGLFormatChannelType(GLFormat::eRG8UI, 0);
					static inline constexpr auto v3 = GetGLFormatChannelSize(GLFormat::eRG8UI, 0);
					static inline constexpr auto v4 = GetGLFormatChannelType(GLFormat::eRG8UI, 2);
					static inline constexpr auto v5 = GetGLFormatChannelSize(GLFormat::eRG8UI, 2);
				};
			}

			struct GLTextureCreateDesc
			{
				GLImageType imageType;
				GLFormat    format;
				GLExtent3D  extent;
				uint32_t    mipLevels;
				uint32_t    arrayLayers;
			};
		}
	}
}

#undef RTLIB_EXT_GL_GL_TYPE_DEF_4
#undef RTLIB_EXT_GL_GL_TYPE_DEF_3
#undef RTLIB_EXT_GL_GL_FORMAT_DEF_1
#undef RTLIB_EXT_GL_GL_FORMAT_DEF_2
#undef RTLIB_EXT_GL_GL_FORMAT_DEF_3
#undef RTLIB_EXT_GL_GL_FORMAT_DEF_4
#endif
