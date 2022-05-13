#ifndef RTLIB_EXT_GL_GL_COMMON_H
#define RTLIB_EXT_GL_GL_COMMON_H
#include <half.h>
#include <glad/glad.h>
#include <cstdint>
#define RTLIB_EXT_GL_GL_FORMAT_DEF_1(VAL1) ((uint64_t)VAL1)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_2(VAL1, VAL2) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_3(VAL1, VAL2, VAL3) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_4(VAL1, VAL2, VAL3, VAL4) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18) | (((uint64_t)VAL4) << 27)

#define RTLIB_EXT_GL_GL_TYPE_DEF_3(VAL1, VAL2, VAL3) ((uint64_t)VAL1 << 10) | (((uint64_t)VAL2) << 16) | (((uint64_t)VAL3) << 32)
#define RTLIB_EXT_GL_GL_TYPE_DEF_4(VAL1, VAL2, VAL3, VAL4) ((uint64_t)VAL1 << 10) | (((uint64_t)VAL2) << 16) | (((uint64_t)VAL3) << 32) | (((uint64_t)VAL4) << 38)

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(VAL1,VAL2) \
	case GL_##VAL1:                                   \
		return GLFormat::e##VAL2

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(VAL1) \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(VAL1, VAL1)

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_SNORM(VAL) \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(VAL##_SNORM,VAL##Snorm)

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_COMPRESS() \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RED,CompressedRed);   \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RG,CompressedRG);    \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGB,CompressedRGB);   \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGBA,CompressedRGBA)

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N(N) \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(R##N);     \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RG##N);    \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGB##N);   \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGBA##N)

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_SNORM(N) \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_SNORM(R##N);     \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_SNORM(RG##N);    \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_SNORM(RGB##N);   \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_SNORM(RGBA##N)

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(N, PREFIX) \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(R##N##PREFIX);            \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RG##N##PREFIX);           \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGB##N##PREFIX);          \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGBA##N##PREFIX)


#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(VAL1,VAL2) \
	case GLFormat::e##VAL2:                                 \
		return GL_##VAL1

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(VAL) \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(VAL,VAL)

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_SNORM_INV(VAL) \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(VAL##_SNORM,VAL##Snorm)

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_COMPRESS_INV() \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RED,CompressedRed);   \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RG,CompressedRG);    \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RGB,CompressedRGB);   \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RGBA,CompressedRGBA)

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_INV(N) \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(R##N);     \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RG##N);    \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGB##N);   \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGBA##N)

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_SNORM_INV(N) \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_SNORM_INV(R##N);     \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_SNORM_INV(RG##N);    \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_SNORM_INV(RGB##N);   \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_SNORM_INV(RGBA##N)

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(N, PREFIX) \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(R##N##PREFIX);            \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RG##N##PREFIX);           \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGB##N##PREFIX);          \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGBA##N##PREFIX)


#define RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(VAL1,VAL2) \
	case GL_##VAL1:                                   \
		return GLTypeFlagBits::e##VAL2

#define RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(VAL1,VAL2) \
	case GLTypeFlagBits::e##VAL2:                                   \
		return GL_##VAL1

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

			struct GLExtent2D
			{
				uint32_t width;
				uint32_t height;
			};
			struct GLExtent3D
			{
				uint32_t width;
				uint32_t height;
				uint32_t depth;
			};

			struct GLOffset2D
			{
				int32_t x;
				int32_t y;
			};
			struct GLOffset3D
			{
				int32_t x;
				int32_t y;
				int32_t z;
			};

			struct GLImageSubresourceLayers
			{
				uint32_t mipLevels;
				uint32_t baseArrayLayer;
				uint32_t layerCount;
			};

			struct GLBufferCopy
			{
				size_t srcOffset;
				size_t dstOffset;
				size_t size;
			};

			struct GLMemoryBufferCopy
			{
				const void *srcData;
				size_t dstOffset;
				size_t size;
			};

			struct GLBufferMemoryCopy
			{
				void *dstData;
				size_t srcOffset;
				size_t size;
			};

			struct GLBufferImageCopy
			{
				size_t bufferOffset;
				size_t bufferSize;
				GLImageSubresourceLayers imageSubresources;
				GLOffset3D imageOffset;
				GLExtent3D imageExtent;
			};

			struct GLMemoryImageCopy
			{
				const void *srcData;
				GLImageSubresourceLayers srcImageSubresources;
				GLOffset3D srcImageOffset;
				GLExtent3D srcImageExtent;
			};

			struct GLImageMemoryCopy
			{
				void *srcData;
				GLImageSubresourceLayers dstImageSubresources;
				GLOffset3D dstImageOffset;
				GLExtent3D dstImageExtent;
			};

			enum GLBufferUsageFlagBits : unsigned int
			{
				GLBufferUsageFlagBitsVertex = 1 << 0,
				GLBufferUsageFlagBitsAtomicCounter = 1 << 1,
				GLBufferUsageFlagBitsDispatchIndirect = 1 << 2,
				GLBufferUsageFlagBitsDrawIndirect = 1 << 3,
				GLBufferUsageFlagBitsIndex = 1 << 4,
				GLBufferUsageFlagBitsImageCopySrc = 1 << 5,
				GLBufferUsageFlagBitsImageCopyDst = 1 << 6,
				GLBufferUsageFlagBitsQuery = 1 << 7,
				GLBufferUsageFlagBitsStorage = 1 << 8,
				GLBufferUsageFlagBitsTexture = 1 << 9,
				GLBufferUsageFlagBitsTransformFeedBack = 1 << 10,
				GLBufferUsageFlagBitsUniform = 1 << 11,
			};
			using GLBufferUsageFlags = GLBufferUsageFlagBits;

			struct GLBufferDesc
			{
				size_t sizeInBytes;
				GLBufferUsageFlags usage;
			};

			enum class GLImageType
			{
				e1D,
				e2D,
				e3D
			};
			///                    1
			//                     Snorm
			//
			enum class GLTypeFlagBits : uint64_t
			{
				eUndefined = 0,
				eInteger = 1 << 6,
				eUnsigned = 1 << 7,
				eFloat = 1 << 8,
				eRev = 1 << 9,

				eInt8 = 8 | eInteger,
				eInt16 = 16 | eInteger,
				eInt32 = 32 | eInteger,

				eUInt8 = 8 | eUnsigned,
				eUInt16 = 16 | eUnsigned,
				eUInt32 = 32 | eUnsigned,

				eFloat16 = 16 | eFloat,
				eFloat32 = 32 | eFloat,

				eUInt8_3_3_2 = eUInt8 | RTLIB_EXT_GL_GL_TYPE_DEF_3(3, 3, 2),
				eUInt8_2_3_3_Rev = eUInt8 | eRev | RTLIB_EXT_GL_GL_TYPE_DEF_3(2, 3, 3),
				eUInt16_4_4_4_4 = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(4, 4, 4, 4),
				eUInt16_4_4_4_4_Rev = eUInt16 | eRev | RTLIB_EXT_GL_GL_TYPE_DEF_4(4, 4, 4, 4),
				eUInt16_5_5_5_1 = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(5, 5, 5, 1),
				eUInt16_1_5_5_5_Rev = eUInt16 | eRev | RTLIB_EXT_GL_GL_TYPE_DEF_4(1, 5, 5, 5),
				eUInt32_8_8_8_8 = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(8, 8, 8, 8),
				eUInt32_8_8_8_8_Rev = eUInt32 | eRev | RTLIB_EXT_GL_GL_TYPE_DEF_4(8, 8, 8, 8),
				eUInt32_10_10_10_2 = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(10, 10, 10, 2),
				eUInt32_2_10_10_10_Rev = eUInt32 | eRev | RTLIB_EXT_GL_GL_TYPE_DEF_4(2, 10, 10, 10),
			};

			inline constexpr auto GetGLenumGLType(GLenum glEnum) -> GLTypeFlagBits
			{
				switch (glEnum)
				{
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(HALF_FLOAT    , Float16);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(FLOAT         , Float32);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_SHORT, UInt16);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_BYTE , UInt8);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_INT  , UInt32);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(BYTE, Int8);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(SHORT,Int16);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(INT,  Int32);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_BYTE_3_3_2, UInt8_3_3_2);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_BYTE_2_3_3_REV, UInt8_2_3_3_Rev);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_SHORT_4_4_4_4, UInt16_4_4_4_4);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_SHORT_4_4_4_4_REV, UInt16_4_4_4_4_Rev);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_SHORT_5_5_5_1, UInt16_5_5_5_1);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_SHORT_1_5_5_5_REV, UInt16_1_5_5_5_Rev);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_INT_8_8_8_8, UInt32_8_8_8_8);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_INT_8_8_8_8_REV, UInt32_8_8_8_8_Rev);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_INT_10_10_10_2, UInt32_10_10_10_2);
				RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_INT_2_10_10_10_REV, UInt32_2_10_10_10_Rev);
				default: return GLTypeFlagBits::eUndefined;
				}
			}

			inline constexpr auto GetGLFormatGLenum(GLTypeFlagBits glType) -> GLenum
			{
				switch (glType)
				{
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(HALF_FLOAT, Float16);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(FLOAT, Float32);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_SHORT, UInt16);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_BYTE, UInt8);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_INT, UInt32);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(BYTE, Int8);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(SHORT, Int16);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(INT, Int32);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_BYTE_3_3_2, UInt8_3_3_2);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_BYTE_2_3_3_REV, UInt8_2_3_3_Rev);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_SHORT_4_4_4_4, UInt16_4_4_4_4);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_SHORT_4_4_4_4_REV, UInt16_4_4_4_4_Rev);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_SHORT_5_5_5_1, UInt16_5_5_5_1);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_SHORT_1_5_5_5_REV, UInt16_1_5_5_5_Rev);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_INT_8_8_8_8, UInt32_8_8_8_8);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_INT_8_8_8_8_REV, UInt32_8_8_8_8_Rev);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_INT_10_10_10_2, UInt32_10_10_10_2);
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_INT_2_10_10_10_REV, UInt32_2_10_10_10_Rev);
				default:
					return 0;
				}
			}
			inline constexpr auto GetGLTypeBaseType(GLTypeFlagBits glType) -> GLTypeFlagBits
			{
				return static_cast<GLTypeFlagBits>(static_cast<uint64_t>(glType) & 511);
			}
			inline constexpr auto GetGLTypeSize(GLTypeFlagBits glType) -> uint32_t
			{
				return static_cast<uint32_t>(static_cast<uint64_t>(glType) & 63);
			}
			inline constexpr auto GetGLTypeChannelSize(GLTypeFlagBits glType, uint32_t channel) -> uint32_t
			{
				return (static_cast<uint64_t>(glType) >> (6 * channel + 10)) & ((1 << 6) - 1);
			}

			template<GLTypeFlagBits typeFlags, bool Cond = GetGLTypeChannelSize(typeFlags,0) == 0>
			struct GLTypeTraits;

			template<>
			struct GLTypeTraits<GLTypeFlagBits::eFloat16> {
				using base_type = half;
				using type      = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eFloat32> {
				using base_type = float;
				using type      = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eUInt8> {
				using base_type = uint8_t;
				using type      = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eUInt16> {
				using base_type = uint16_t;
				using type = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eUInt32> {
				using base_type = uint32_t;
				using type      = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eInt8> {
				using base_type = int8_t;
				using type = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eInt16> {
				using base_type = int16_t;
				using type = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eInt32> {
				using base_type = int32_t;
				using type = base_type;
			};

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

			enum class GLFormat : uint64_t
			{
				//
				eUndefined = 0,
				eInteger = 1 << 6,
				eUnsigned = 1 << 7,
				eFloat = 1 << 8,

				eRed = ((uint64_t)1) << 32,
				eGreen = ((uint64_t)1) << 33,
				eBlue = ((uint64_t)1) << 34,
				eAlpha = ((uint64_t)1) << 35,
				eDepth = ((uint64_t)1) << 36,
				eStencil = ((uint64_t)1) << 37,
				eSnorm = ((uint64_t)1) << 38,
				eSharedbit = ((uint64_t)1) << 39,
				eBaseInteger = ((uint64_t)1) << 40,

				eBaseRed = eRed,
				eBaseRG = eRed | eGreen,
				eBaseRGB = eRed | eGreen | eBlue,
				eBaseRGBA = eRed | eGreen | eBlue | eAlpha,

				eBaseIntegerRed = eBaseRed | eBaseInteger,
				eBaseIntegerRG = eBaseRG | eBaseInteger,
				eBaseIntegerRGB = eBaseRGB | eBaseInteger,
				eBaseIntegerRGBA = eBaseRGBA | eBaseInteger,

				eBaseDepth = eDepth,
				eBaseDepthStencil = eDepth | eStencil,

				eSRGB = (((uint64_t)1) << 41) | eBaseRGB,
				eSRGBA = (((uint64_t)1) << 41) | eBaseRGBA,
				eCompressed = ((uint64_t)1) << 42,
				eRGTC1 = ((uint64_t)1) << 43,
				eSignedRGTC1 = ((uint64_t)1) << 44,
				eRGTC2 = ((uint64_t)1) << 45,
				eSignedRGTC2 = ((uint64_t)1) << 46,
				eBPTCUnorm = ((uint64_t)1) << 47,
				eBPTCSignedFloat = ((uint64_t)1) << 48,
				eBPTCUnsignedFloat = ((uint64_t)1) << 49,

				eCompressedRed = eCompressed | eBaseRed,
				eCompressedRG = eCompressed | eBaseRG,
				eCompressedRGB = eCompressed | eBaseRGB,
				eCompressedRGBA = eCompressed | eBaseRGBA,
				eCompressedSRGB = eCompressed | eSRGB,
				eCompressedSRGBA = eCompressed | eSRGBA,

				eCompressedRedRGTC1 = eCompressedRed | eRGTC1,
				eCompressedSignedRedRGTC1 = eCompressedRed | eSignedRGTC1,

				eCompressedRGRGTC2 = eCompressedRG | eRGTC2,
				eCompressedSignedRGRGTC2 = eCompressedRG | eSignedRGTC2,

				eCompressedRGBABPTCUnorm = eCompressedRGBA|eBPTCUnorm,
				eCompressedSRGBABPTCUnorm= eCompressedSRGBA|eBPTCUnorm,

				eCompressedSRGBBPTCSignedFloat = eCompressedRGB | eBPTCSignedFloat,
				eCompressedSRGBBPTCUnsignedFloat = eCompressedRGB | eBPTCUnsignedFloat,

				// RGBA
				eR8 = eBaseRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(8),
				eR8Snorm = eBaseRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(8) | eSnorm,
				eR16 = eBaseRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16),
				eR16Snorm = eBaseRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16) | eSnorm,
				eRG8 = eBaseRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(8, 8),
				eRG8Snorm = eBaseRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(8, 8) | eSnorm,
				eRG16 = eBaseRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(16, 16),
				eRG16Snorm = eBaseRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(16, 16) | eSnorm,
				eR3G3B2 = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(3, 3, 2),
				eRGB4 = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(4, 4, 4),
				eRGB5 = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(5, 5, 5),
				eRGB8 = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(8, 8, 8),
				eRGB8Snorm = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(8, 8, 8) | eSnorm,
				eRGB10 = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(10, 10, 10),
				eRGB12 = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(12, 12, 12),
				eRGB16 = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(16, 16, 16),
				eRGB16Snorm = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(16, 16, 16) | eSnorm,
				eRGBA2 = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(2, 2, 2, 2),
				eRGBA4 = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(4, 4, 4, 4),
				eRGB5A1 = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(5, 5, 5, 1),
				eRGBA8 = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(8, 8, 8, 8),
				eRGBA8Snorm = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(8, 8, 8, 8) | eSnorm,
				eRGB10A2 = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(10, 10, 10, 2),
				eRGB10A2UI = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(10 | eUnsigned, 10 | eUnsigned, 10 | eUnsigned, 2 | eUnsigned),
				eRGBA12 = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(12, 12, 12, 12),
				eRGBA16 = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(16, 16, 16, 16),
				eRGBA16Snorm = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(16, 16, 16, 16) | eSnorm,
				eSRGB8 = eSRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(8, 8, 8),
				eSRGBA8 = eSRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(8, 8, 8, 8),
				eRGB565 = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(5, 6, 5),
				eR16F = eBaseRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16 | eFloat),
				eRG16F = eBaseRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(16 | eFloat, 16 | eFloat),
				eRGB16F = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(16 | eFloat, 16 | eFloat, 16 | eFloat),
				eRGBA16F = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(16 | eFloat, 16 | eFloat, 16 | eFloat, 16 | eFloat),
				eR32F = eBaseRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(32 | eFloat),
				eRG32F = eBaseRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(32 | eFloat, 32 | eFloat),
				eRGB32F = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(32 | eFloat, 32 | eFloat, 32 | eFloat),
				eRGBA32F = eBaseRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(32 | eFloat, 32 | eFloat, 32 | eFloat, 32 | eFloat),
				eR11FG11FB10F = eBaseRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(11 | eFloat, 11 | eFloat, 10 | eFloat),
				eRGB9E5 = eBaseRGB | eSharedbit | RTLIB_EXT_GL_GL_FORMAT_DEF_4(9, 9, 9, 5),
				eR8UI = eBaseIntegerRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(8 | eUnsigned),
				eRG8UI = eBaseIntegerRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(8 | eUnsigned, 8 | eUnsigned),
				eRGB8UI = eBaseIntegerRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(8 | eUnsigned, 8 | eUnsigned, 8 | eUnsigned),
				eRGBA8UI = eBaseIntegerRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(8 | eUnsigned, 8 | eUnsigned, 8 | eUnsigned, 8 | eUnsigned),
				eR8I = eBaseIntegerRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(8 | eInteger),
				eRG8I = eBaseIntegerRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(8 | eInteger, 8 | eInteger),
				eRGB8I = eBaseIntegerRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(8 | eInteger, 8 | eInteger, 8 | eInteger),
				eRGBA8I = eBaseIntegerRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(8 | eInteger, 8 | eInteger, 8 | eInteger, 8 | eInteger),
				eR16I = eBaseIntegerRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16 | eInteger),
				eRG16I = eBaseIntegerRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(16 | eInteger, 16 | eInteger),
				eRGB16I = eBaseIntegerRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(16 | eInteger, 16 | eInteger, 16 | eInteger),
				eRGBA16I = eBaseIntegerRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(16 | eInteger, 16 | eInteger, 16 | eInteger, 16 | eInteger),
				eR16UI = eBaseIntegerRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16 | eUnsigned),
				eRG16UI = eBaseIntegerRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(16 | eUnsigned, 16 | eUnsigned),
				eRGB16UI = eBaseIntegerRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(16 | eUnsigned, 16 | eUnsigned, 16 | eUnsigned),
				eRGBA16UI = eBaseIntegerRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(16 | eUnsigned, 16 | eUnsigned, 16 | eUnsigned, 16 | eUnsigned),
				eR32I = eBaseIntegerRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(32 | eInteger),
				eRG32I = eBaseIntegerRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(32 | eInteger, 32 | eInteger),
				eRGB32I = eBaseIntegerRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(32 | eInteger, 32 | eInteger, 32 | eInteger),
				eRGBA32I = eBaseIntegerRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(32 | eInteger, 32 | eInteger, 32 | eInteger, 32 | eInteger),
				eR32UI = eBaseIntegerRed | RTLIB_EXT_GL_GL_FORMAT_DEF_1(32 | eUnsigned),
				eRG32UI = eBaseIntegerRG | RTLIB_EXT_GL_GL_FORMAT_DEF_2(32 | eUnsigned, 32 | eUnsigned),
				eRGB32UI = eBaseIntegerRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(32 | eUnsigned, 32 | eUnsigned, 32 | eUnsigned),
				eRGBA32UI = eBaseIntegerRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(32 | eUnsigned, 32 | eUnsigned, 32 | eUnsigned, 32 | eUnsigned),
				// Depth
				eDepth16 = eBaseDepth | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16),
				eDepth24 = eBaseDepth | RTLIB_EXT_GL_GL_FORMAT_DEF_1(24),
				eDepth32F = eBaseDepth | RTLIB_EXT_GL_GL_FORMAT_DEF_1(32 | eFloat),
				// DepthStencil
				eDepth24Stencil8 = eBaseDepthStencil | RTLIB_EXT_GL_GL_FORMAT_DEF_2(24, 8),
				eDepth32FStencil8 = eBaseDepthStencil | RTLIB_EXT_GL_GL_FORMAT_DEF_2(32 | eFloat, 8),
			};

			inline constexpr auto GetGLenumGLFormat(GLenum glEnum) -> GLFormat
			{
				switch (glEnum)
				{
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_COMPRESS();
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_SRGB, CompressedSRGB);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_SRGB_ALPHA, CompressedSRGBA);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RED_RGTC1, CompressedRedRGTC1);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_SIGNED_RED_RGTC1, CompressedSignedRedRGTC1);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RG_RGTC2, CompressedRGRGTC2);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_SIGNED_RG_RGTC2, CompressedSignedRGRGTC2);

					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGBA_BPTC_UNORM,  CompressedRGBABPTCUnorm);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_SRGB_ALPHA_BPTC_UNORM, CompressedSRGBABPTCUnorm);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGB_BPTC_SIGNED_FLOAT, CompressedSRGBBPTCSignedFloat);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, CompressedSRGBBPTCUnsignedFloat);
					
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(R3_G3_B2, R3G3B2);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(RGB5_A1 , RGB5A1);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGB565);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(RGB9_E5,RGB9E5);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(SRGB8, SRGB8);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(SRGB8_ALPHA8, SRGBA8);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(RGB10_A2, RGB10A2);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(RGB10_A2UI, RGB10A2UI);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(R11F_G11F_B10F, R11FG11FB10F);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGB4);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGB5);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGB10);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGB12);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGBA2);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGBA4);

					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N(8);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_SNORM(8);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(8, I);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(8, UI);

					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N(16);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_SNORM(16);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(16, I);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(16, UI);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(16, F);

					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(32, I);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(32, UI);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX(32, F);
				default:
					return GLFormat::eUndefined;
				}
			}

			inline constexpr auto GetGLFormatGLenum(GLFormat format) -> GLenum
			{
				switch (format)
				{
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_COMPRESS_INV();
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_SRGB, CompressedSRGB);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_SRGB_ALPHA, CompressedSRGBA);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RED_RGTC1, CompressedRedRGTC1);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_SIGNED_RED_RGTC1, CompressedSignedRedRGTC1);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RG_RGTC2, CompressedRGRGTC2);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_SIGNED_RG_RGTC2, CompressedSignedRGRGTC2);

					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RGBA_BPTC_UNORM, CompressedRGBABPTCUnorm);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_SRGB_ALPHA_BPTC_UNORM, CompressedSRGBABPTCUnorm);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RGB_BPTC_SIGNED_FLOAT, CompressedSRGBBPTCSignedFloat);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, CompressedSRGBBPTCUnsignedFloat);

					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(R3_G3_B2, R3G3B2);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(RGB5_A1, RGB5A1);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGB565);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(RGB9_E5, RGB9E5);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(SRGB8, SRGB8);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(SRGB8_ALPHA8, SRGBA8);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(RGB10_A2, RGB10A2);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(RGB10_A2UI, RGB10A2UI);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(R11F_G11F_B10F, R11FG11FB10F);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGB4);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGB5);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGB10);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGB12);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGBA2);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(RGBA4);

					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_INV(8);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_SNORM_INV(8);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(8, I);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(8, UI);

					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_INV(16);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_SNORM_INV(16);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(16, I);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(16, UI);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(16, F);

					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(32, I);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(32, UI);
					RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV(32, F);
				default:
					return 0;
				}
			}

			inline constexpr auto GetGLFormatBaseFormat(GLFormat format) -> GLFormat
			{
				return static_cast<GLFormat>(static_cast<uint64_t>(format) & (static_cast<uint64_t>(GLFormat::eBaseRGBA) | static_cast<uint64_t>(GLFormat::eBaseDepthStencil) | static_cast<uint64_t>(GLFormat::eBaseInteger)));
			}

			inline constexpr auto GetGLFormatChannelSize(GLFormat format, uint32_t channel) -> uint32_t
			{
				return (static_cast<uint64_t>(format) >> (9 * channel)) & ((1 << 6) - 1);
			}

			inline constexpr auto GetGLFormatChannelType(GLFormat format, uint32_t channel) -> GLFormat
			{
				return static_cast<GLFormat>((static_cast<uint64_t>(format) >> (9 * channel)) & (((1 << 3) - 1) << 6));
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
			

			struct GLTextureDesc
			{
				GLImageType imageType;
				GLFormat format;
				GLExtent3D extent;
				uint32_t mipLevels;
				uint32_t arrayLayers;
			};
		}
	}
}

#undef RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2 
#undef RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE
#undef RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_SNORM
#undef RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_COMPRESS
#undef RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N
#undef RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_SNORM
#undef RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_N_PREFIX
#undef RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV
#undef RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV
#undef RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_SNORM_INV
#undef RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_COMPRESS_INV
#undef RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_INV
#undef RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_SNORM_INV
#undef RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_N_PREFIX_INV
#undef RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2
#undef RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV
#undef RTLIB_EXT_GL_GL_TYPE_DEF_4
#undef RTLIB_EXT_GL_GL_TYPE_DEF_3
#undef RTLIB_EXT_GL_GL_FORMAT_DEF_1
#undef RTLIB_EXT_GL_GL_FORMAT_DEF_2
#undef RTLIB_EXT_GL_GL_FORMAT_DEF_3
#undef RTLIB_EXT_GL_GL_FORMAT_DEF_4
#endif
