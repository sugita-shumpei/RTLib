#ifndef RTLIB_EXT_GL_GL_TYPE_CONVERSIONS_H
#define RTLIB_EXT_GL_GL_TYPE_CONVERSIONS_H
#include <RTLib/Ext/GL/GLCommon.h>
#define RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(VAL1,VAL2) \
	case GL_##VAL1:                                   \
		return GLTypeFlagBits::e##VAL2

#define RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(VAL1,VAL2) \
	case GLTypeFlagBits::e##VAL2:                                   \
		return GL_##VAL1

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

namespace RTLib {
	namespace Ext {
		namespace GL
		{
			inline constexpr auto GetGLBufferMainUsageTarget(GLBufferUsageFlags usageFlags)->GLenum
			{
				if (usageFlags & GLBufferUsageVertex           ) {
					return GL_ARRAY_BUFFER;
				}
				if (usageFlags & GLBufferUsageIndex            ) {
					return GL_ELEMENT_ARRAY_BUFFER;
				}
				if (usageFlags & GLBufferUsageUniform          ) {
					return GL_UNIFORM_BUFFER;
				}
				if (usageFlags & GLBufferUsageStorage          ) {
					return GL_SHADER_STORAGE_BUFFER;
				}
				if (usageFlags & GLBufferUsageImageCopySrc     ) {
					return GL_PIXEL_UNPACK_BUFFER;
				}
				if (usageFlags & GLBufferUsageImageCopyDst     ) {
					return GL_PIXEL_PACK_BUFFER;
				}
				if (usageFlags & GLBufferUsageDrawIndirect     ) {
					return GL_DRAW_INDIRECT_BUFFER;
				}
				if (usageFlags & GLBufferUsageDispatchIndirect ) {
					return GL_DISPATCH_INDIRECT_BUFFER;
				}
				if (usageFlags & GLBufferUsageTransformFeedBack) {
					return GL_TRANSFORM_FEEDBACK_BUFFER;
				}
				if (usageFlags & GLBufferUsageTexture          ) {
					return GL_TEXTURE_BUFFER;
				}
				if (usageFlags & GLBufferUsageQuery            ) {
					return GL_QUERY_BUFFER;
				}
				if (usageFlags & GLBufferUsageAtomicCounter    ) {
					return GL_ATOMIC_COUNTER_BUFFER;
				}
				return GL_ARRAY_BUFFER;
			}

			inline constexpr auto GetGLenumGLType(GLenum glEnum) -> GLTypeFlagBits
			{
				switch (glEnum)
				{
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(HALF_FLOAT, Float16);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(FLOAT, Float32);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_SHORT, UInt16);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_BYTE, UInt8);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_INT, UInt32);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(BYTE, Int8);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(SHORT, Int16);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(INT, Int32);
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

					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGBA_BPTC_UNORM, CompressedRGBABPTCUnorm);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_SRGB_ALPHA_BPTC_UNORM, CompressedSRGBABPTCUnorm);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGB_BPTC_SIGNED_FLOAT, CompressedSRGBBPTCSignedFloat);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, CompressedSRGBBPTCUnsignedFloat);

					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(R3_G3_B2, R3G3B2);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(RGB5_A1, RGB5A1);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(RGB565);
					RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(RGB9_E5, RGB9E5);
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

			inline constexpr auto GetGLMemoryPropertyFlagsGLAccessFlags(GLMemoryPropertyFlags flags)->GLenum
			{
				GLenum res = 0;
				if (flags == GLMemoryPropertyDeviceLocal) {
					return 0;
				}
				if (flags & GLMemoryPropertyHostVisible) {
					res |= GL_CLIENT_STORAGE_BIT;
					res |= GL_MAP_READ_BIT ;
					res |= GL_MAP_WRITE_BIT;
				}
				if (flags & GLMemoryPropertyHostCoherent) {
					res |= GL_MAP_COHERENT_BIT;
				}
				return res;
			}
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
#endif
