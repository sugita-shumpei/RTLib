#ifndef RTLIB_EXT_GL_GL_TYPE_FORMAT_H
#define RTLIB_EXT_GL_GL_TYPE_FORMAT_H
#include <half.h>
#define RTLIB_EXT_GL_GL_FORMAT_DEF_1(VAL1) ((uint64_t)VAL1)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_2(VAL1, VAL2) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_3(VAL1, VAL2, VAL3) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_4(VAL1, VAL2, VAL3, VAL4) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18) | (((uint64_t)VAL4) << 27)

#define RTLIB_EXT_GL_GL_TYPE_DEF_3(VAL1, VAL2, VAL3) ((uint64_t)VAL1 << 10) | (((uint64_t)VAL2) << 16) | (((uint64_t)VAL3) << 32)
#define RTLIB_EXT_GL_GL_TYPE_DEF_4(VAL1, VAL2, VAL3, VAL4) ((uint64_t)VAL1 << 10) | (((uint64_t)VAL2) << 16) | (((uint64_t)VAL3) << 32) | (((uint64_t)VAL4) << 38)
#include <RTLib/Core/TypeFormat.h>
namespace RTLib {
	namespace Ext {
		namespace GL {
			enum class GLTypeFlagBits : uint64_t
			{
				eUndefined = 0,

				eInt8 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8),
				eInt16 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16),
				eInt32 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32),

				eUInt8 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8),
				eUInt16 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16),
				eUInt32 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32),

				eFloat16 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16),
				eFloat32 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32),

				eUInt8_3_3_2 = eUInt8 | RTLIB_EXT_GL_GL_TYPE_DEF_3(3, 3, 2),
				eUInt8_2_3_3_Rev = eUInt8 | RTLIB_EXT_GL_GL_TYPE_DEF_3(2, 3, 3) | (1 << 9),
				eUInt16_4_4_4_4 = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(4, 4, 4, 4) | (1 << 9),
				eUInt16_4_4_4_4_Rev = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(4, 4, 4, 4),
				eUInt16_5_5_5_1 = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(5, 5, 5, 1),
				eUInt16_1_5_5_5_Rev = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(1, 5, 5, 5) | (1 << 9),
				eUInt32_8_8_8_8 = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(8, 8, 8, 8),
				eUInt32_8_8_8_8_Rev = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(8, 8, 8, 8) | (1 << 9),
				eUInt32_10_10_10_2 = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(10, 10, 10, 2),
				eUInt32_2_10_10_10_Rev = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(2, 10, 10, 10) | (1 << 9),
			};

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

			template<GLTypeFlagBits typeFlags, bool Cond = GetGLTypeChannelSize(typeFlags, 0) == 0>
			struct GLTypeTraits;

			template<>
			struct GLTypeTraits<GLTypeFlagBits::eFloat16> {
				using base_type = half;
				using type = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eFloat32> {
				using base_type = float;
				using type = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eUInt8> {
				using base_type = uint8_t;
				using type = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eUInt16> {
				using base_type = uint16_t;
				using type = base_type;
			};
			template<>
			struct GLTypeTraits<GLTypeFlagBits::eUInt32> {
				using base_type = uint32_t;
				using type = base_type;
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
			enum class GLAttachmentComponentExt :uint64_t {
				eSnorm = (((uint64_t)1) << 38),
				eSharedbit = (((uint64_t)1) << 39),
				eBaseInteger = (((uint64_t)1) << 40),
				eSrgbNonlinear = (((uint64_t)1) << 41),
				eCompressed = (((uint64_t)1) << 42),
				eRGTC1 = (((uint64_t)1) << 43),
				eSignedRGTC1 = (((uint64_t)1) << 44),
				eRGTC2 = (((uint64_t)1) << 45),
				eSignedRGTC2 = (((uint64_t)1) << 46),
				eBPTCUnorm = (((uint64_t)1) << 47),
				eBPTCSignedFloat = (((uint64_t)1) << 48),
				eBPTCUnsignedFloat = (((uint64_t)1) << 49),
			};

			enum class GLBaseFormat : uint64_t
			{
				eBaseRed = static_cast<uint64_t>(Core::BaseFormat::eBaseRed),
				eBaseRG = static_cast<uint64_t>(Core::BaseFormat::eBaseRG),
				eBaseRGB = static_cast<uint64_t>(Core::BaseFormat::eBaseRGB),
				eBaseRGBA = static_cast<uint64_t>(Core::BaseFormat::eBaseRGBA),

				eBaseIntegerRed = eBaseRed | static_cast<uint64_t>(GLAttachmentComponentExt::eBaseInteger),
				eBaseIntegerRG = eBaseRG | static_cast<uint64_t>(GLAttachmentComponentExt::eBaseInteger),
				eBaseIntegerRGB = eBaseRGB | static_cast<uint64_t>(GLAttachmentComponentExt::eBaseInteger),
				eBaseIntegerRGBA = eBaseRGBA | static_cast<uint64_t>(GLAttachmentComponentExt::eBaseInteger),

				eBaseDepth = static_cast<uint64_t>(Core::BaseFormat::eBaseDepth),
				eBaseStencil = static_cast<uint64_t>(Core::BaseFormat::eBaseStencil),
				eBaseDepthStencil = static_cast<uint64_t>(Core::BaseFormat::eBaseDepthStencil),
			};

			enum class GLFormat : uint64_t
			{
				//
				eUndefined = 0,
				eSRGB = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | static_cast<uint64_t>(GLAttachmentComponentExt::eSrgbNonlinear),
				eSRGBA = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | static_cast<uint64_t>(GLAttachmentComponentExt::eSrgbNonlinear),

				eCompressedRed = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | static_cast<uint64_t>(GLAttachmentComponentExt::eCompressed),
				eCompressedRG = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | static_cast<uint64_t>(GLAttachmentComponentExt::eCompressed),
				eCompressedRGB = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | static_cast<uint64_t>(GLAttachmentComponentExt::eCompressed),
				eCompressedRGBA = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | static_cast<uint64_t>(GLAttachmentComponentExt::eCompressed),
				eCompressedSRGB = eSRGB | static_cast<uint64_t>(GLAttachmentComponentExt::eCompressed),
				eCompressedSRGBA = eSRGBA | static_cast<uint64_t>(GLAttachmentComponentExt::eCompressed),

				eCompressedRedRGTC1 = eCompressedRed | static_cast<uint64_t>(GLAttachmentComponentExt::eRGTC1),
				eCompressedSignedRedRGTC1 = eCompressedRed | static_cast<uint64_t>(GLAttachmentComponentExt::eSignedRGTC1),

				eCompressedRGRGTC2 = eCompressedRG | static_cast<uint64_t>(GLAttachmentComponentExt::eRGTC2),
				eCompressedSignedRGRGTC2 = eCompressedRG | static_cast<uint64_t>(GLAttachmentComponentExt::eSignedRGTC2),

				eCompressedRGBABPTCUnorm = eCompressedRGBA | static_cast<uint64_t>(GLAttachmentComponentExt::eBPTCUnorm),
				eCompressedSRGBABPTCUnorm = eCompressedSRGBA | static_cast<uint64_t>(GLAttachmentComponentExt::eBPTCUnorm),

				eCompressedSRGBBPTCSignedFloat = eCompressedRGB | static_cast<uint64_t>(GLAttachmentComponentExt::eBPTCSignedFloat),
				eCompressedSRGBBPTCUnsignedFloat = eCompressedRGB | static_cast<uint64_t>(GLAttachmentComponentExt::eBPTCUnsignedFloat),

				// RGBA
				eR8 = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(8),
				eR8Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(8) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eR16 = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16),
				eR16Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRG8 = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(8, 8),
				eRG8Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(8, 8) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRG16 = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(16, 16),
				eRG16Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(16, 16) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eR3G3B2 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(3, 3, 2),
				eRGB4 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(4, 4, 4),
				eRGB5 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(5, 5, 5),
				eRGB8 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(8, 8, 8),
				eRGB8Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(8, 8, 8) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRGB10 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(10, 10, 10),
				eRGB12 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(12, 12, 12),
				eRGB16 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(16, 16, 16),
				eRGB16Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(16, 16, 16) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRGBA2 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(2, 2, 2, 2),
				eRGBA4 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(4, 4, 4, 4),
				eRGB5A1 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(5, 5, 5, 1),
				eRGBA8 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(8, 8, 8, 8),
				eRGBA8Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(8, 8, 8, 8) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRGB10A2 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(10, 10, 10, 2),
				eRGB10A2UI = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(10 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned), 10 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned), 10 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned), 2 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned)),
				eRGBA12 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(12, 12, 12, 12),
				eRGBA16 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(16, 16, 16, 16),
				eRGBA16Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(16, 16, 16, 16) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eSRGB8 = eSRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_3(8, 8, 8),
				eSRGBA8 = eSRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_4(8, 8, 8, 8),
				eRGB565 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(5, 6, 5),
				eR16F = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16)),
				eRG16F = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16)),
				eRGB16F = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16)),
				eRGBA16F = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16)),
				eR32F = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				eRG32F = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				eRGB32F = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				eRGBA32F = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				eR11FG11FB10F = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(11 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eFloat), 11 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eFloat), 10 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eFloat)),
				eRGB9E5 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(9, 9, 9, 5) | static_cast<uint64_t>(GLAttachmentComponentExt::eSharedbit),
				eR8UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8)),
				eRG8UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8)),
				eRGB8UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8)),
				eRGBA8UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8)),
				eR8I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8)),
				eRG8I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8)),
				eRGB8I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8)),
				eRGBA8I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8)),
				eR16UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16)),
				eRG16UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16)),
				eRGB16UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16)),
				eRGBA16UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16)),
				eR16I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16)),
				eRG16I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16)),
				eRGB16I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16)),
				eRGBA16I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16)),
				eR32UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32)),
				eRG32UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32)),
				eRGB32UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32)),
				eRGBA32UI = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32)),
				eR32I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32)),
				eRG32I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32)),
				eRGB32I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32)),
				eRGBA32I = static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32)),
				// Depth
				eDepth16 = static_cast<uint64_t>(GLBaseFormat::eBaseDepth) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(16),
				eDepth24 = static_cast<uint64_t>(GLBaseFormat::eBaseDepth) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(24),
				eDepth32F = static_cast<uint64_t>(GLBaseFormat::eBaseDepth) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				// DepthStencil
				eDepth24Stencil8 = static_cast<uint64_t>(GLBaseFormat::eBaseDepthStencil) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(24, 8),
				eDepth32FStencil8 = static_cast<uint64_t>(GLBaseFormat::eBaseDepthStencil) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), 8),
			};

			inline constexpr auto GetGLFormatBaseFormat(GLFormat format) -> GLBaseFormat
			{
				return static_cast<GLBaseFormat>(static_cast<uint64_t>(format) & (static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGBA) | static_cast<uint64_t>(GLBaseFormat::eBaseDepthStencil)));
			}
			inline constexpr auto GetGLFormatChannelSize(GLFormat format, uint32_t channel) -> uint32_t
			{
				return (static_cast<uint64_t>(format) >> (9 * channel)) & ((1 << 6) - 1);
			}
			inline constexpr auto GetGLFormatChannelType(GLFormat format, uint32_t channel) -> GLFormat
			{
				return static_cast<GLFormat>((static_cast<uint64_t>(format) >> (9 * channel)) & (((1 << 3) - 1) << 6));
			}
		}
	}
}

#endif