#ifndef RTLIB_EXT_GL_GL_TYPE_CONVERSIONS_H
#define RTLIB_EXT_GL_GL_TYPE_CONVERSIONS_H
#include <RTLib/Ext/GL/GLCommon.h>
#define RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(VAL1, VAL2) \
	case GL_##VAL1:                                         \
		return GLTypeFlagBits::e##VAL2

#define RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(VAL1, VAL2) \
	case GLTypeFlagBits::e##VAL2:                               \
		return GL_##VAL1

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(VAL1, VAL2) \
	case GL_##VAL1:                                           \
		return GLFormat::e##VAL2

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE(VAL1) \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(VAL1, VAL1)

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_SNORM(VAL) \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(VAL##_SNORM, VAL##Snorm)

#define RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_M_COMPRESS()                  \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RED, CompressedRed); \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RG, CompressedRG);   \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGB, CompressedRGB); \
	RTLIB_EXT_GL_GET_GL_ENUM_GL_FORMAT_CASE_2(COMPRESSED_RGBA, CompressedRGBA)

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

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(VAL1, VAL2) \
	case GLFormat::e##VAL2:                                       \
		return GL_##VAL1

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_INV(VAL) \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(VAL, VAL)

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_SNORM_INV(VAL) \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(VAL##_SNORM, VAL##Snorm)

#define RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_M_COMPRESS_INV()                  \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RED, CompressedRed); \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RG, CompressedRG);   \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RGB, CompressedRGB); \
	RTLIB_EXT_GL_GET_GL_FORMAT_GL_ENUM_CASE_2_INV(COMPRESSED_RGBA, CompressedRGBA)

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

namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			inline constexpr auto GetGLBufferMainUsageTarget(GLBufferUsageFlags usageFlags) -> GLenum
			{
				if (usageFlags & GLBufferUsageVertex)
				{
					return GL_ARRAY_BUFFER;
				}
				if (usageFlags & GLBufferUsageIndex)
				{
					return GL_ELEMENT_ARRAY_BUFFER;
				}
				if (usageFlags & GLBufferUsageUniform)
				{
					return GL_UNIFORM_BUFFER;
				}
				if (usageFlags & GLBufferUsageStorage)
				{
					return GL_SHADER_STORAGE_BUFFER;
				}
				if (usageFlags & GLBufferUsageImageCopySrc)
				{
					return GL_PIXEL_UNPACK_BUFFER;
				}
				if (usageFlags & GLBufferUsageImageCopyDst)
				{
					return GL_PIXEL_PACK_BUFFER;
				}
				if (usageFlags & GLBufferUsageDrawIndirect)
				{
					return GL_DRAW_INDIRECT_BUFFER;
				}
				if (usageFlags & GLBufferUsageDispatchIndirect)
				{
					return GL_DISPATCH_INDIRECT_BUFFER;
				}
				if (usageFlags & GLBufferUsageTransformFeedBack)
				{
					return GL_TRANSFORM_FEEDBACK_BUFFER;
				}
				if (usageFlags & GLBufferUsageTexture)
				{
					return GL_TEXTURE_BUFFER;
				}
				if (usageFlags & GLBufferUsageQuery)
				{
					return GL_QUERY_BUFFER;
				}
				if (usageFlags & GLBufferUsageAtomicCounter)
				{
					return GL_ATOMIC_COUNTER_BUFFER;
				}
				if (usageFlags & GLBufferUsageGenericCopyDst)
				{
					return GL_COPY_WRITE_BUFFER;
				}
				if (usageFlags & GLBufferUsageGenericCopySrc)
				{
					return GL_COPY_READ_BUFFER;
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
					RTLIB_EXT_GL_GET_GL_ENUM_GL_TYPE_CASE_2(UNSIGNED_INT_10F_11F_11F_REV, UInt32_10F_11F_11F_Rev);
				default:
					return GLTypeFlagBits::eUndefined;
				}
			}
			inline constexpr auto GetGLTypeGLEnum(GLTypeFlagBits glType) -> GLenum
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
					RTLIB_EXT_GL_GET_GL_TYPE_GL_ENUM_CASE_2_INV(UNSIGNED_INT_10F_11F_11F_REV, UInt32_10F_11F_11F_Rev);
				default:
					return GL_FLOAT;
				};
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

			inline constexpr auto GetGLBaseFormatGLenum(GLBaseFormat baseFormat) -> GLenum
			{
				switch (baseFormat)
				{
				case RTLib::Ext::GL::GLBaseFormat::eBaseRed:
					return GL_RED;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseRG:
					return GL_RG;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseRGB:
					return GL_RGB;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseRGBA:
					return GL_RGBA;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseIntegerRed:
					return GL_RED_INTEGER;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseIntegerRG:
					return GL_RG_INTEGER;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseIntegerRGB:
					return GL_RGB_INTEGER;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseIntegerRGBA:
					return GL_RGBA_INTEGER;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseDepth:
					return GL_DEPTH;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseStencil:
					return GL_STENCIL;
					break;
				case RTLib::Ext::GL::GLBaseFormat::eBaseDepthStencil:
					return GL_DEPTH_STENCIL;
					break;
				default:
					return GL_RED;
					break;
				}
			}

			inline constexpr auto GetGLFormatGLUnpackEnum(GLFormat format) -> GLenum
			{
				switch (format)
				{

				case RTLib::Ext::GL::GLFormat::eUndefined:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eSRGB:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eSRGBA:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRed:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRG:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRGB:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eCompressedRGBA:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGB:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGBA:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRedRGTC1:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSignedRedRGTC1:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRGRGTC2:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSignedRGRGTC2:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRGBABPTCUnorm:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGBABPTCUnorm:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGBBPTCSignedFloat:
					return GL_UNSIGNED_BYTE;
					break;
					// TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGBBPTCUnsignedFloat:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eR8:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eR8Snorm:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eR16:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eR16Snorm:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRG8:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRG8Snorm:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16Snorm:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eR3G3B2:
					return GL_UNSIGNED_BYTE_3_3_2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB4:
					return GL_UNSIGNED_SHORT_4_4_4_4;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB5:
					return GL_UNSIGNED_SHORT_5_5_5_1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB8:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB8Snorm:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB10:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB12:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16Snorm:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA2:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA4:
					return GL_UNSIGNED_SHORT_4_4_4_4;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB5A1:
					return GL_UNSIGNED_SHORT_5_5_5_1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA8:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA8Snorm:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB10A2:
					return GL_UNSIGNED_INT_10_10_10_2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB10A2UI:
					return GL_UNSIGNED_INT_10_10_10_2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA12:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16Snorm:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eSRGB8:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eSRGBA8:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB565:
					return GL_UNSIGNED_SHORT_5_6_5;
					break;
				case RTLib::Ext::GL::GLFormat::eR16F:
					return GL_HALF_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16F:
					return GL_HALF_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16F:
					return GL_HALF_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16F:
					return GL_HALF_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eR32F:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eRG32F:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB32F:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA32F:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eR11FG11FB10F:
					return GL_UNSIGNED_INT_10F_11F_11F_REV;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB9E5:
					return GL_UNSIGNED_INT_5_9_9_9_REV;
					break;
				case RTLib::Ext::GL::GLFormat::eR8UI:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRG8UI:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB8UI:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA8UI:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eR8I:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRG8I:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB8I:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA8I:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLFormat::eR16UI:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16UI:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16UI:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16UI:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eR16I:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16I:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16I:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16I:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eR32UI:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eRG32UI:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB32UI:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA32UI:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eR32I:
					return GL_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eRG32I:
					return GL_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB32I:
					return GL_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA32I:
					return GL_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth16:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth24:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth32F:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth24Stencil8:
					return GL_UNSIGNED_INT_24_8;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth32FStencil8:
					return GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
					break;
				default:
					break;
				}
			}

			inline constexpr auto GetGLFormatGLUnpackCount(GLFormat format)->uint32_t
			{
				switch (format)
				{
				case RTLib::Ext::GL::GLFormat::eUndefined: return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eSRGB: return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eSRGBA: return 4;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRed:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRG:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eCompressedRGB:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRGBA:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGB:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGBA:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRedRGTC1:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSignedRedRGTC1:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRGRGTC2:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSignedRGRGTC2:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedRGBABPTCUnorm:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGBABPTCUnorm:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGBBPTCSignedFloat:return 1;
					break;
					//TODO
				case RTLib::Ext::GL::GLFormat::eCompressedSRGBBPTCUnsignedFloat:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eR8: return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eR8Snorm:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eR16:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eR16Snorm:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG8:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRG8Snorm:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16Snorm:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eR3G3B2:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB4:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB5:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB8:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB8Snorm:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB10:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB12:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16Snorm:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA2:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA4:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB5A1:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA8:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA8Snorm:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB10A2:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB10A2UI:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA12:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16Snorm:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eSRGB8:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eSRGBA8:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB565:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eR16F:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16F:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16F:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16F:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eR32F:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG32F:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB32F:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA32F:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eR11FG11FB10F:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB9E5:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eR8UI:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG8UI:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB8UI:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA8UI:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eR8I:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG8I:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB8I:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA8I:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eR16UI:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16UI:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16UI:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16UI:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eR16I:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG16I:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB16I:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA16I:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eR32UI:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG32UI:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB32UI:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA32UI:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eR32I:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eRG32I:return 2;
					break;
				case RTLib::Ext::GL::GLFormat::eRGB32I:return 3;
					break;
				case RTLib::Ext::GL::GLFormat::eRGBA32I:return 4;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth16:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth24:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth32F:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth24Stencil8:return 1;
					break;
				case RTLib::Ext::GL::GLFormat::eDepth32FStencil8:return 1;
					break;
				default:
					break;
				}
			}

			inline constexpr auto GetGLVertexFormatGLenum(GLVertexFormat format) -> GLenum
			{
				switch (format)
				{
				case RTLib::Ext::GL::GLVertexFormat::eInt8x1:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt8x2:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt8x3:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt8x4:
					return GL_BYTE;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt8x1:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt8x2:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt8x3:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt8x4:
					return GL_UNSIGNED_BYTE;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt16x1:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt16x2:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt16x3:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt16x4:
					return GL_SHORT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt16x1:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt16x2:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt16x3:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt16x4:
					return GL_UNSIGNED_SHORT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt32x1:
					return GL_INT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt32x2:
					return GL_INT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt32x3:
					return GL_INT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt32x4:
					return GL_INT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32x1:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32x2:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32x3:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32x4:
					return GL_UNSIGNED_INT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat16x1:
					return GL_HALF_FLOAT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat16x2:
					return GL_HALF_FLOAT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat16x3:
					return GL_HALF_FLOAT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat16x4:
					return GL_HALF_FLOAT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat32x1:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat32x2:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat32x3:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat32x4:
					return GL_FLOAT;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32_2_10_10_10_Rev:
					return GL_UNSIGNED_INT_2_10_10_10_REV;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32_10_10_10_2:
					return GL_UNSIGNED_INT_10_10_10_2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32_10F_11F_11F_Rev:
					return GL_UNSIGNED_INT_10F_11F_11F_REV;
					break;
				default:
					return GL_BYTE;
					break;
				}
			}

			inline constexpr auto GetGLVertexFormatNumChannels(GLVertexFormat format) -> uint32_t
			{
				switch (format)
				{
				case RTLib::Ext::GL::GLVertexFormat::eInt8x1:
					return 1;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt8x2:
					return 2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt8x3:
					return 3;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt8x4:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt8x1:
					return 1;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt8x2:
					return 2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt8x3:
					return 3;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt8x4:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt16x1:
					return 1;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt16x2:
					return 2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt16x3:
					return 3;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt16x4:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt16x1:
					return 1;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt16x2:
					return 2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt16x3:
					return 3;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt16x4:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt32x1:
					return 1;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt32x2:
					return 2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt32x3:
					return 3;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eInt32x4:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32x1:
					return 1;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32x2:
					return 2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32x3:
					return 3;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32x4:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat16x1:
					return 1;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat16x2:
					return 2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat16x3:
					return 3;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat16x4:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat32x1:
					return 1;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat32x2:
					return 2;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat32x3:
					return 3;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eFloat32x4:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32_2_10_10_10_Rev:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32_10_10_10_2:
					return 4;
					break;
				case RTLib::Ext::GL::GLVertexFormat::eUInt32_10F_11F_11F_Rev:
					return 3;
					break;
				default:
					return 1;
					break;
				}
			}

			inline constexpr auto GetGLBufferUsageFlagsAccessMode(const GLMemoryPropertyFlags &access) -> GLenum
			{
				if (access == GLMemoryPropertyDefault)
				{
					return 0;
				}
				GLenum res = 0;
				if (access & GLMemoryPropertyHostRead)
				{
					res |= GL_MAP_READ_BIT;
				}
				if (access & GLMemoryPropertyHostWrite)
				{
					res |= GL_MAP_WRITE_BIT;
				}
				if (access & GLMemoryPropertyHostCoherent)
				{
					res |= GL_MAP_COHERENT_BIT;
				}
				if (access & GLMemoryPropertyHostPaged)
				{
					res |= GL_CLIENT_STORAGE_BIT;
				}
				return res;
			}

			inline constexpr auto GetGLBufferCreateDescBufferAccessFrequency(const GLBufferCreateDesc &desc) -> GLenum
			{
				if (desc.usage == GLBufferUsageGenericCopySrc)
				{ // BufferCopySource
					if (desc.access & GLMemoryPropertyHostRead)
					{ // Hostが読みとる
						// Readback
						return GL_STATIC_READ;
					}
				}
				if (desc.usage == GLBufferUsageGenericCopyDst)
				{ // BufferCopyDest
					if (desc.access == GLMemoryPropertyHostPaged)
					{
						return GL_STATIC_COPY;
					}
				}
				return GL_STATIC_DRAW;
			}

			inline constexpr auto GetGLShaderStagesGLShaderType(const GLShaderStageFlagBits shaderStage) -> GLenum
			{
				switch (shaderStage)
				{
				case GLShaderStageVertex:
					return GL_VERTEX_SHADER;
				case GLShaderStageGeometry:
					return GL_GEOMETRY_SHADER;
				case GLShaderStageTessControl:
					return GL_TESS_CONTROL_SHADER;
				case GLShaderStageTessEvaluation:
					return GL_TESS_EVALUATION_SHADER;
				case GLShaderStageFragment:
					return GL_FRAGMENT_SHADER;
				case GLShaderStageCompute:
					return GL_COMPUTE_SHADER;
				default:
					return GL_VERTEX_SHADER;
				}
			}

			inline constexpr auto GetGLShaderStagesGLShaderBits(const GLShaderStageFlagBits shaderStage) -> GLenum
			{
				switch (shaderStage)
				{
					GLenum res = 0;
					if (shaderStage & GLShaderStageVertex)
					{
						res |= GL_VERTEX_SHADER_BIT;
					}
					if (shaderStage & GLShaderStageGeometry)
					{
						res |= GL_GEOMETRY_SHADER;
					}
					if (shaderStage & GLShaderStageTessControl)
					{
						res |= GL_TESS_CONTROL_SHADER;
					}
					if (shaderStage & GLShaderStageTessEvaluation)
					{
						res |= GL_TESS_EVALUATION_SHADER;
					}
					if (shaderStage & GLShaderStageFragment)
					{
						res |= GL_FRAGMENT_SHADER;
					}
					if (shaderStage & GLShaderStageCompute)
					{
						res |= GL_COMPUTE_SHADER;
					}
					return res;
				}
			}

			inline constexpr auto GetGLDrawModeGLenum(const GLDrawMode drawMode) -> GLenum
			{
				switch (drawMode)
				{
				case GLDrawMode::eLines:
					return GL_LINES;
				case GLDrawMode::eLinesAdjacency:
					return GL_LINE_STRIP_ADJACENCY;
				case GLDrawMode::eLineStrip:
					return GL_LINE_STRIP;
				case GLDrawMode::eLineStripAdjacency:
					return GL_LINE_STRIP_ADJACENCY;
				case GLDrawMode::eLineLoop:
					return GL_LINE_LOOP;
				case GLDrawMode::ePoints:
					return GL_POINTS;
				case GLDrawMode::eTriangleFan:
					return GL_TRIANGLE_FAN;
				case GLDrawMode::eTriangles:
					return GL_TRIANGLES;
				case GLDrawMode::eTrianglesAdjacency:
					return GL_TRIANGLES_ADJACENCY;
				case GLDrawMode::eTriangleStrip:
					return GL_TRIANGLE_STRIP;
				case GLDrawMode::eTriangleStripAdjacency:
					return GL_TRIANGLE_STRIP_ADJACENCY;
				default:
					return GL_LINES;
					break;
				}
			}

			inline constexpr auto GetGLMagFilterEnum(Core::FilterMode filterMode) -> GLenum
			{
				switch (filterMode)
				{
				case RTLib::Core::FilterMode::eNearest:
					return GL_NEAREST;
					break;
				case RTLib::Core::FilterMode::eLinear:
					return GL_LINEAR;
					break;
				default:
					break;
				}
			}
			inline constexpr auto GetGLMinFilterEnum(Core::FilterMode filterMode, Core::SamplerMipmapMode mipmapFilterMode) -> GLenum
			{
				switch (filterMode)
				{
				case RTLib::Core::FilterMode::eNearest:
					switch (mipmapFilterMode)
					{
					case RTLib::Core::SamplerMipmapMode::eNearest:
						return GL_NEAREST_MIPMAP_NEAREST;
						break;
					case RTLib::Core::SamplerMipmapMode::eLinear:
						return GL_NEAREST_MIPMAP_LINEAR;
						break;
					default:
						return GL_NEAREST;
						break;
					}
					break;
				case RTLib::Core::FilterMode::eLinear:
					switch (mipmapFilterMode)
					{
					case RTLib::Core::SamplerMipmapMode::eNearest:
						return GL_LINEAR_MIPMAP_NEAREST;
						break;
					case RTLib::Core::SamplerMipmapMode::eLinear:
						return GL_LINEAR_MIPMAP_LINEAR;
						break;
					default:
						return GL_LINEAR;
						break;
					}
					break;
				default:
					return GL_LINEAR;
					break;
				}
			}

			inline constexpr auto GetGLAddressModeGLEnum(Core::SamplerAddressMode mode) -> GLenum
			{
				switch (mode)
				{
				case RTLib::Core::SamplerAddressMode::eRepeat:
					return GL_REPEAT;
					break;
				case RTLib::Core::SamplerAddressMode::eMirroredRepeat:
					return GL_MIRRORED_REPEAT;
					break;
				case RTLib::Core::SamplerAddressMode::eClampToEdge:
					return GL_CLAMP_TO_EDGE;
					break;
				case RTLib::Core::SamplerAddressMode::eClampToBorder:
					return GL_CLAMP_TO_BORDER;
					break;
				case RTLib::Core::SamplerAddressMode::eMirrorClampToEdge:
					return GL_MIRROR_CLAMP_TO_EDGE;
					break;
				default:
					return GL_REPEAT;
					break;
				}
			}

			inline constexpr auto GetGLCompareOpGLEnum(Core::CompareOp compareOp) -> GLenum
			{
				switch (compareOp)
				{
				case RTLib::Core::CompareOp::eNever:
					return GL_NEVER;
					break;
				case RTLib::Core::CompareOp::eLess:
					return GL_LESS;
					break;
				case RTLib::Core::CompareOp::eEqual:
					return GL_EQUAL;
					break;
				case RTLib::Core::CompareOp::eLessOrEqual:
					return GL_LEQUAL;
					break;
				case RTLib::Core::CompareOp::eGreater:
					return GL_GREATER;
					break;
				case RTLib::Core::CompareOp::eNotEqual:
					return GL_NOTEQUAL;
					break;
				case RTLib::Core::CompareOp::eGreaterOrEqual:
					return GL_GEQUAL;
					break;
				case RTLib::Core::CompareOp::eAlways:
					return GL_ALWAYS;
					break;
				default:
					return GL_NEVER;
					break;
				}
			}
			
			inline constexpr auto GetGLImageViewTypeGLenum(GL::GLImageViewType viewType)->GLenum
			{
				switch (viewType)
				{
				case RTLib::Ext::GL::GLImageViewType::e1D: return GL_TEXTURE_1D;
					break;
				case RTLib::Ext::GL::GLImageViewType::e2D: return GL_TEXTURE_2D;
					break;
				case RTLib::Ext::GL::GLImageViewType::e3D: return GL_TEXTURE_3D;
					break;
				case RTLib::Ext::GL::GLImageViewType::eCubemap:return GL_TEXTURE_CUBE_MAP;
					break;
				case RTLib::Ext::GL::GLImageViewType::e1DArray:return GL_TEXTURE_1D_ARRAY;
					break;
				case RTLib::Ext::GL::GLImageViewType::e2DArray:return GL_TEXTURE_2D_ARRAY;
					break;
				case RTLib::Ext::GL::GLImageViewType::eCubemapArray:return GL_TEXTURE_CUBE_MAP_ARRAY;
					break;
				default:return GL_TEXTURE_2D;
					break;
				}
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
