#ifndef RTLIB_EXT_GL_GL_TYPE_FORMAT_H
#define RTLIB_EXT_GL_GL_TYPE_FORMAT_H
#include <half.h>
#define RTLIB_EXT_GL_GL_TYPE_REV_FLAGS (((uint64_t)1)<<9)
//OK
#define RTLIB_EXT_GL_GL_FORMAT_DEF_1(VAL1) ((uint64_t)VAL1)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_2(VAL1, VAL2) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_3(VAL1, VAL2, VAL3) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18)
#define RTLIB_EXT_GL_GL_FORMAT_DEF_4(VAL1, VAL2, VAL3, VAL4) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18) | (((uint64_t)VAL4) << 27)

#define RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_1(SIZE1) RTLIB_EXT_GL_GL_FORMAT_DEF_1((SIZE1-1))
#define RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_2(SIZE1,SIZE2) RTLIB_EXT_GL_GL_FORMAT_DEF_2((SIZE1-1),(SIZE2-1))
#define RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(SIZE1,SIZE2,SIZE3) RTLIB_EXT_GL_GL_FORMAT_DEF_3((SIZE1-1),(SIZE2-1),(SIZE3-1))
#define RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(SIZE1,SIZE2,SIZE3,SIZE4) RTLIB_EXT_GL_GL_FORMAT_DEF_4((SIZE1-1),(SIZE2-1),(SIZE3-1),(SIZE4-1))

#define RTLIB_EXT_GL_GL_TYPE_DEF_3(VAL1, VAL2, VAL3) ((uint64_t)VAL1 << 10) | (((uint64_t)VAL2) << 16) | (((uint64_t)VAL3) << 22)
#define RTLIB_EXT_GL_GL_TYPE_DEF_4(VAL1, VAL2, VAL3, VAL4) ((uint64_t)VAL1 << 10) | (((uint64_t)VAL2) << 16) | (((uint64_t)VAL3) << 22) | (((uint64_t)VAL4) << 28)
#include <RTLib/Core/TypeFormat.h>
namespace RTLib {
	namespace Ext {
		namespace GL {
			enum class GLBaseTypeFlagBits : uint64_t
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
				eFloat64 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat64),
			};
			enum class GLDataTypeFlagBits : uint64_t
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
				eFloat64 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat64),

				eUInt8_3_3_2           = eUInt8  | RTLIB_EXT_GL_GL_TYPE_DEF_3(3, 3, 2),
				eUInt8_2_3_3_Rev       = eUInt8  | RTLIB_EXT_GL_GL_TYPE_DEF_3(2, 3, 3)       | RTLIB_EXT_GL_GL_TYPE_REV_FLAGS,
				eUInt16_4_4_4_4        = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(4, 4, 4, 4)    | RTLIB_EXT_GL_GL_TYPE_REV_FLAGS,
				eUInt16_4_4_4_4_Rev    = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(4, 4, 4, 4),
				eUInt16_5_5_5_1        = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(5, 5, 5, 1),
				eUInt16_1_5_5_5_Rev    = eUInt16 | RTLIB_EXT_GL_GL_TYPE_DEF_4(1, 5, 5, 5)    | RTLIB_EXT_GL_GL_TYPE_REV_FLAGS,
				eUInt32_8_8_8_8        = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(8, 8, 8, 8),
				eUInt32_8_8_8_8_Rev    = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(8, 8, 8, 8)    | RTLIB_EXT_GL_GL_TYPE_REV_FLAGS,
				eUInt32_10_10_10_2     = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(10, 10, 10, 2),
				eUInt32_2_10_10_10_Rev = eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_4(2, 10, 10, 10) | RTLIB_EXT_GL_GL_TYPE_REV_FLAGS,
				eUInt32_10F_11F_11F_Rev= eUInt32 | RTLIB_EXT_GL_GL_TYPE_DEF_3(10, 11, 11)    | RTLIB_EXT_GL_GL_TYPE_REV_FLAGS,
			};
			struct   GLDataTypeFlagsUtils
			{
				inline static constexpr bool IsDerived(GLDataTypeFlagBits glType)noexcept {
					return static_cast<uint64_t>(glType) >> 10;
				}
				inline static constexpr auto GetBaseType(GLDataTypeFlagBits glType) -> GLDataTypeFlagBits
				{
					return static_cast<GLDataTypeFlagBits>(static_cast<uint64_t>(glType) & 511);
				}
				inline static constexpr auto GetBaseTypeBitSize(GLDataTypeFlagBits glType) -> uint32_t
				{
					return Core::SizedTypeFlagsUtils:: GetTypeBitSize(static_cast<Core::SizedTypeFlagBits>(GetBaseType(glType)));
				}
				inline static constexpr auto GetDerivedChannelBitSize( GLDataTypeFlagBits glType, uint32_t channel) -> uint32_t
				{
					return (static_cast<uint64_t>(glType) >> (6 * channel + 10)) & ((1 << 6) - 1);
				}
				inline static constexpr bool HasDerivedChannel(GLDataTypeFlagBits glType, uint32_t channel) {
					return GetDerivedChannelBitSize(glType, channel)>0;
				}
				inline static constexpr auto GetDerivedNumChannels(GLDataTypeFlagBits glType) -> unsigned int {
					if (GetDerivedChannelBitSize(glType, 3) > 0) { return 4; }
					if (GetDerivedChannelBitSize(glType, 2) > 0) { return 3; }
					if (GetDerivedChannelBitSize(glType, 1) > 0) { return 2; }
					if (GetDerivedChannelBitSize(glType, 0) > 0) { return 1; }
					return 0;
				}
			};
			namespace {
				static inline constexpr auto v1 = GLDataTypeFlagsUtils::GetDerivedChannelBitSize(GLDataTypeFlagBits::eUInt32_10_10_10_2, 0);
				static inline constexpr auto v2 = GLDataTypeFlagsUtils::GetDerivedChannelBitSize(GLDataTypeFlagBits::eUInt32_10_10_10_2, 1);
				static inline constexpr auto v3 = GLDataTypeFlagsUtils::GetDerivedChannelBitSize(GLDataTypeFlagBits::eUInt32_10_10_10_2, 2);
				static inline constexpr auto v4 = GLDataTypeFlagsUtils::GetDerivedChannelBitSize(GLDataTypeFlagBits::eUInt32_10_10_10_2, 3);
				static_assert(v1 == 10);
				static_assert(v2 == 10);
				static_assert(v3 == 10);
				static_assert(v4 == 2);
				static inline constexpr auto v5 = GLDataTypeFlagsUtils::GetDerivedChannelBitSize(GLDataTypeFlagBits::eUInt32_10F_11F_11F_Rev, 0);
				static inline constexpr auto v6 = GLDataTypeFlagsUtils::GetDerivedChannelBitSize(GLDataTypeFlagBits::eUInt32_10F_11F_11F_Rev, 1);
				static inline constexpr auto v7 = GLDataTypeFlagsUtils::GetDerivedChannelBitSize(GLDataTypeFlagBits::eUInt32_10F_11F_11F_Rev, 2);
				static inline constexpr auto v8 = GLDataTypeFlagsUtils::GetDerivedChannelBitSize(GLDataTypeFlagBits::eUInt32_10F_11F_11F_Rev, 3);
				static_assert(v5 == 10);
				static_assert(v6 == 11);
				static_assert(v7 == 11);
				static_assert(v8 ==  0);
				static inline constexpr auto v9  = GLDataTypeFlagsUtils::IsDerived(GLDataTypeFlagBits::eUInt8_3_3_2);
				static inline constexpr auto v10 = GLDataTypeFlagsUtils::IsDerived(GLDataTypeFlagBits::eUInt16_4_4_4_4);
				static inline constexpr auto v11 = GLDataTypeFlagsUtils::IsDerived(GLDataTypeFlagBits::eUInt32_8_8_8_8_Rev);
				static inline constexpr auto v12 = GLDataTypeFlagsUtils::IsDerived(GLDataTypeFlagBits::eUInt32_10_10_10_2);
				static inline constexpr auto v13 = GLDataTypeFlagsUtils::IsDerived(GLDataTypeFlagBits::eInt8);
				static inline constexpr auto v14 = GLDataTypeFlagsUtils::IsDerived(GLDataTypeFlagBits::eInt16);
				static inline constexpr auto v15 = GLDataTypeFlagsUtils::IsDerived(GLDataTypeFlagBits::eInt32);
				static_assert( v9 && v10&& v11&& v12);
				static_assert(!v13&&!v14&&!v15);
				static inline constexpr auto v16 = GLDataTypeFlagsUtils::GetDerivedNumChannels(GLDataTypeFlagBits::eUInt8_3_3_2);
				static inline constexpr auto v17 = GLDataTypeFlagsUtils::GetDerivedNumChannels(GLDataTypeFlagBits::eUInt16_4_4_4_4);
				static inline constexpr auto v18 = GLDataTypeFlagsUtils::GetDerivedNumChannels(GLDataTypeFlagBits::eUInt32_8_8_8_8_Rev);
				static inline constexpr auto v19 = GLDataTypeFlagsUtils::GetDerivedNumChannels(GLDataTypeFlagBits::eUInt32_10_10_10_2);
				static inline constexpr auto v20 = GLDataTypeFlagsUtils::GetDerivedNumChannels(GLDataTypeFlagBits::eInt8);
				static inline constexpr auto v21 = GLDataTypeFlagsUtils::GetDerivedNumChannels(GLDataTypeFlagBits::eInt16);
				static inline constexpr auto v22 = GLDataTypeFlagsUtils::GetDerivedNumChannels(GLDataTypeFlagBits::eInt32);
				static_assert(v16 == 3);
				static_assert(v17 == 4);
				static_assert(v18 == 4);
				static_assert(v19 == 4);
				static_assert(v20 == 0);
				static_assert(v21 == 0);
				static_assert(v22 == 0);
			};

			template<GLBaseTypeFlagBits baseFlags>
			struct GLBaseTypeTraits;
			template<>
			struct GLBaseTypeTraits<GLBaseTypeFlagBits::eFloat16> {
				using base_type = half;
				using type = base_type;
			};
			template<>
			struct GLBaseTypeTraits<GLBaseTypeFlagBits::eFloat32> {
				using base_type = float;
				using type = base_type;
			};
			template<>
			struct GLBaseTypeTraits<GLBaseTypeFlagBits::eUInt8> {
				using base_type = uint8_t;
				using type = base_type;
			};
			template<>
			struct GLBaseTypeTraits<GLBaseTypeFlagBits::eUInt16> {
				using base_type = uint16_t;
				using type = base_type;
			};
			template<>
			struct GLBaseTypeTraits<GLBaseTypeFlagBits::eUInt32> {
				using base_type = uint32_t;
				using type = base_type;
			};
			template<>
			struct GLBaseTypeTraits<GLBaseTypeFlagBits::eInt8> {
				using base_type = int8_t;
				using type = base_type;
			};
			template<>
			struct GLBaseTypeTraits<GLBaseTypeFlagBits::eInt16> {
				using base_type = int16_t;
				using type = base_type;
			};
			template<>
			struct GLBaseTypeTraits<GLBaseTypeFlagBits::eInt32> {
				using base_type = int32_t;
				using type = base_type;
			};


			template<GLDataTypeFlagBits typeFlags, bool Cond = GLDataTypeFlagsUtils::GetDerivedNumChannels(typeFlags) == 0 >
			struct GLDataTypeTraits;
			template<>
			struct GLDataTypeTraits<GLDataTypeFlagBits::eFloat16> {
				using base_type = half;
				using type = base_type;
			};
			template<>
			struct GLDataTypeTraits<GLDataTypeFlagBits::eFloat32> {
				using base_type = float;
				using type = base_type;
			};
			template<>
			struct GLDataTypeTraits<GLDataTypeFlagBits::eUInt8> {
				using base_type = uint8_t;
				using type = base_type;
			};
			template<>
			struct GLDataTypeTraits<GLDataTypeFlagBits::eUInt16> {
				using base_type = uint16_t;
				using type = base_type;
			};
			template<>
			struct GLDataTypeTraits<GLDataTypeFlagBits::eUInt32> {
				using base_type = uint32_t;
				using type = base_type;
			};
			template<>
			struct GLDataTypeTraits<GLDataTypeFlagBits::eInt8> {
				using base_type = int8_t;
				using type = base_type;
			};
			template<>
			struct GLDataTypeTraits<GLDataTypeFlagBits::eInt16> {
				using base_type = int16_t;
				using type = base_type;
			};
			template<>
			struct GLDataTypeTraits<GLDataTypeFlagBits::eInt32> {
				using base_type = int32_t;
				using type = base_type;
			};

			using GLAttachmentComponent = Core::AttachmentCompponent;
			enum class GLAttachmentComponentExt :uint64_t {
				eSnorm = (((uint64_t)1) << 42),
				eSharedbit = (((uint64_t)1) << 43),
				eBaseInteger = (((uint64_t)1) << 44),
				eSrgbNonlinear = (((uint64_t)1) << 45),
				eCompressed = (((uint64_t)1) << 46),
				eRGTC1 = (((uint64_t)1) << 47),
				eSignedRGTC1 = (((uint64_t)1) << 48),
				eRGTC2 = (((uint64_t)1) << 49),
				eSignedRGTC2 = (((uint64_t)1) << 50),
				eBPTCUnorm = (((uint64_t)1) << 51),
				eBPTCSignedFloat= (((uint64_t)1) << 52),
				eBPTCUnsignedFloat = (((uint64_t)1) << 53),
			};

			enum class GLBaseFormat : uint64_t
			{
				eBaseRed  = static_cast<uint64_t>(Core::BaseFormat::eBaseRed),
				eBaseRG   = static_cast<uint64_t>(Core::BaseFormat::eBaseRG),
				eBaseRGB  = static_cast<uint64_t>(Core::BaseFormat::eBaseRGB),
				eBaseRGBA = static_cast<uint64_t>(Core::BaseFormat::eBaseRGBA),

				eBaseIntegerRed  = eBaseRed  | static_cast<uint64_t>(GLAttachmentComponentExt::eBaseInteger),
				eBaseIntegerRG   = eBaseRG   | static_cast<uint64_t>(GLAttachmentComponentExt::eBaseInteger),
				eBaseIntegerRGB  = eBaseRGB  | static_cast<uint64_t>(GLAttachmentComponentExt::eBaseInteger),
				eBaseIntegerRGBA = eBaseRGBA | static_cast<uint64_t>(GLAttachmentComponentExt::eBaseInteger),

				eBaseDepth        = static_cast<uint64_t>(Core::BaseFormat::eBaseDepth),
				eBaseStencil      = static_cast<uint64_t>(Core::BaseFormat::eBaseStencil),
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
				eR8 = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_1(8),
				eR8Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_1(8) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eR16 = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_1(16),
				eR16Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_1(16) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRG8 = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_2(8, 8),
				eRG8Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_2(8, 8) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRG16 = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_2(16, 16),
				eRG16Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_2(16, 16) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eR3G3B2 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(3, 3, 2),
				eRGB4 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(4, 4, 4),
				eRGB5 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(5, 5, 5),
				eRGB8 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(8, 8, 8),
				eRGB8Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(8, 8, 8) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRGB10 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(10, 10, 10),
				eRGB12 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(12, 12, 12),
				eRGB16 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(16, 16, 16),
				eRGB16Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(16, 16, 16) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRGBA2 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(2, 2, 2, 2),
				eRGBA4 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(4, 4, 4, 4),
				eRGB5A1 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(5, 5, 5, 1),
				eRGBA8 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(8, 8, 8, 8),
				eRGBA8Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(8, 8, 8, 8) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eRGB10A2 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(10, 10, 10, 2),
				eRGB10A2UI = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(9 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned), 9 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned), 9 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned), 1 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned)),
				eRGBA12 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(12, 12, 12, 12),
				eRGBA16 = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(16, 16, 16, 16),
				eRGBA16Snorm = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(16, 16, 16, 16) | static_cast<uint64_t>(GLAttachmentComponentExt::eSnorm),
				eSRGB8  = eSRGB | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(8, 8, 8),
				eSRGBA8 = eSRGBA | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(8, 8, 8, 8),
				eRGB565 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_3(5, 6, 5),
				eR16F = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16)),
				eRG16F = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16)),
				eRGB16F = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16)),
				eRGBA16F = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16)),
				eR32F = static_cast<uint64_t>(GLBaseFormat::eBaseRed) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				eRG32F = static_cast<uint64_t>(GLBaseFormat::eBaseRG) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				eRGB32F = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				eRGBA32F = static_cast<uint64_t>(GLBaseFormat::eBaseRGBA) | RTLIB_EXT_GL_GL_FORMAT_DEF_4(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				eR11FG11FB10F = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_3(10 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eFloat), 10 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eFloat), 9 | static_cast<uint64_t>(Core::BaseTypeFlagBits::eFloat)),
				eRGB9E5 = static_cast<uint64_t>(GLBaseFormat::eBaseRGB) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_4(9, 9, 9, 5) | static_cast<uint64_t>(GLAttachmentComponentExt::eSharedbit),
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
				eDepth16 = static_cast<uint64_t>(GLBaseFormat::eBaseDepth) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_1(16),
				eDepth24 = static_cast<uint64_t>(GLBaseFormat::eBaseDepth) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_1(24),
				eDepth32F = static_cast<uint64_t>(GLBaseFormat::eBaseDepth) | RTLIB_EXT_GL_GL_FORMAT_DEF_1(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32)),
				// DepthStencil
				eDepth24Stencil8 = static_cast<uint64_t>(GLBaseFormat::eBaseDepthStencil) | RTLIB_EXT_GL_GL_FORMAT_DEF_SIZE_2(24, 8),
				eDepth32FStencil8 = static_cast<uint64_t>(GLBaseFormat::eBaseDepthStencil) | RTLIB_EXT_GL_GL_FORMAT_DEF_2(static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32), 7),
			};

			struct GLFormatUtils
			{
				static constexpr bool IsComressed(GLFormat format) {
					return static_cast<uint64_t>(format) & static_cast<uint64_t>(GLAttachmentComponentExt::eCompressed);
				}
				static constexpr auto GetBaseFormat(GLFormat format) -> GLBaseFormat
				{
					return static_cast<GLBaseFormat>(static_cast<uint64_t>(format) & (static_cast<uint64_t>(GLBaseFormat::eBaseIntegerRGBA) | static_cast<uint64_t>(GLBaseFormat::eBaseDepthStencil)));
				}
				static constexpr auto GetDerivedChannelType(GLFormat format, uint32_t channel)->Core::BaseTypeFlagBits
				{
					return static_cast<Core::BaseTypeFlagBits>((static_cast<uint64_t>(format) >> (9 * channel)) & (((1 << 3) - 1) << 6));
				}
				static constexpr auto GetDerivedChannelBitSize(GLFormat format, uint32_t channel)->uint32_t
				{
					auto v = (static_cast<uint32_t>((static_cast<uint64_t>(format) >> (9 * channel)) & ((1 << 6) - 1)));
					return v> 0 ?v+1:0;
				}
				static constexpr bool HasDerivedChannel(GLFormat format, uint32_t channel) {
					return GetDerivedChannelBitSize(format, channel) > 0;
				}
				static constexpr auto GetDerivedNumChannels(GLFormat format)->uint32_t {
					if (HasDerivedChannel(format, 3)) { return 4; }
					if (HasDerivedChannel(format, 2)) { return 3; }
					if (HasDerivedChannel(format, 1)) { return 2; }
					if (HasDerivedChannel(format, 0)) { return 1; }
					return 0;
				}
				static constexpr auto GetDerivedBitSize(GLFormat format)->uint32_t {
					if (format == GLFormat::eDepth32FStencil8) { return 64; }
					auto channel0 = GetDerivedChannelBitSize(format, 0);
					if (channel0 == 0) { return 0; }
					auto channel1 = GetDerivedChannelBitSize(format, 1);
					if (channel1 == 0) { return channel0; }
					auto channel2 = GetDerivedChannelBitSize(format, 2);
					if (channel2 == 0) { return channel0+ channel1; }
					auto channel3 = GetDerivedChannelBitSize(format, 3);
					if (channel3 == 0) { return channel0 + channel1 + channel2; }
					return channel0+channel1+channel2+channel3;
				}
			};
			struct TestGLFormat
			{
				static inline constexpr bool v0[10] = {
					GLBaseFormat::eBaseRGBA         == GLFormatUtils::GetBaseFormat(GLFormat::eRGBA8),
					GLBaseFormat::eBaseRGB          == GLFormatUtils::GetBaseFormat(GLFormat::eR11FG11FB10F),
					GLBaseFormat::eBaseDepthStencil == GLFormatUtils::GetBaseFormat(GLFormat::eDepth32FStencil8),
					GLBaseFormat::eBaseRGBA         == GLFormatUtils::GetBaseFormat(GLFormat::eRGB10A2UI),
				};
				static inline constexpr bool v1[10] = {
					8 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGBA8, 0),
					8 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGBA8, 1),
					8 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGBA8, 2),
					8 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGBA8, 3),
				};
				static inline constexpr bool v2[30] = {
					8 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGB8, 0),
					8 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGB8, 1),
					8 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGB8, 2),
					0 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGB8, 3),
					11== GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eR11FG11FB10F, 0),
					11== GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eR11FG11FB10F, 1),
					10== GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eR11FG11FB10F, 2),
					0 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eR11FG11FB10F, 3),
					32== GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGBA32UI, 0),
					32== GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGBA32UI, 1),
					32== GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGBA32UI, 2),
					32== GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGBA32UI, 3),
					5 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGB565, 0),
					6 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGB565, 1),
					5 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGB565, 2),
					0 == GLFormatUtils::GetDerivedChannelBitSize(GLFormat::eRGB565, 3),
				};
				static inline constexpr bool v3[10] = {
					3 == GLFormatUtils::GetDerivedNumChannels(GLFormat::eR11FG11FB10F),
					2 == GLFormatUtils::GetDerivedNumChannels(GLFormat::eDepth32FStencil8),
					4 == GLFormatUtils::GetDerivedNumChannels(GLFormat::eRGB10A2UI),
				};
				static inline constexpr Core::BaseTypeFlagBits v4[10] = {
					GLFormatUtils::GetDerivedChannelType(GLFormat::eRGBA32UI,0),
					GLFormatUtils::GetDerivedChannelType(GLFormat::eRGBA32UI,1),
					GLFormatUtils::GetDerivedChannelType(GLFormat::eRGBA32UI,2),
					GLFormatUtils::GetDerivedChannelType(GLFormat::eRGBA32UI,3),

					GLFormatUtils::GetDerivedChannelType(GLFormat::eRGB565,0),
					GLFormatUtils::GetDerivedChannelType(GLFormat::eRGB565,1),
					GLFormatUtils::GetDerivedChannelType(GLFormat::eRGB565,2),
					GLFormatUtils::GetDerivedChannelType(GLFormat::eRGB565,3),
				};
				static inline constexpr bool v5[20] = {
					GLFormatUtils::GetDerivedBitSize(GLFormat::eRGBA32UI)==128,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eDepth32FStencil8)==64,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eRGBA32UI)==128,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eRGBA32UI) == 128,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eRGBA16) == 64,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eRGBA8) == 32,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eRGB565) == 16,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eR11FG11FB10F) == 32,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eRGB9E5)==32,
					GLFormatUtils::GetDerivedBitSize(GLFormat::eRGB10A2UI)== 32,
				};
			};
			enum class GLVertexFormat : uint64_t
			{
				//GL_BYTE
				eInt8x1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X1),
				eInt8x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X2),
				eInt8x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X3),
				eInt8x4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X4),
				//GL_UNSIGNED_BYTE
				eUInt8x1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X1),
				eUInt8x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X2),
				eUInt8x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X3),
				eUInt8x4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X4),
				//GL_SHORT
				eInt16x1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X1),
				eInt16x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X2),
				eInt16x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X3),
				eInt16x4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X4),
				//GL_UNSIGNED_SHORT
				eUInt16x1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X1),
				eUInt16x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X2),
				eUInt16x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X3),
				eUInt16x4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X4),
				//GL_INT
				eInt32x1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X1),
				eInt32x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X2),
				eInt32x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X3),
				eInt32x4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X4),
				//GL_UNSIGNED_INT
				eUInt32x1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X1),
				eUInt32x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X2),
				eUInt32x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X3),
				eUInt32x4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X4),
				//GL_INT
				eFloat16x1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X1),
				eFloat16x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X2),
				eFloat16x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X3),
				eFloat16x4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X4),
				//GL_UNSIGNED_INT
				eFloat32x1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X1),
				eFloat32x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X2),
				eFloat32x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X3),
				eFloat32x4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X4),
				//
				eUInt32_2_10_10_10_Rev = static_cast<uint64_t>(GL::GLDataTypeFlagBits::eUInt32_2_10_10_10_Rev),
				eUInt32_10_10_10_2     = static_cast<uint64_t>(GL::GLDataTypeFlagBits::eUInt32_10_10_10_2),
				eUInt32_10F_11F_11F_Rev= static_cast<uint64_t>(GL::GLDataTypeFlagBits::eUInt32_10F_11F_11F_Rev),
			};
			enum class GLIndexFormat :uint64_t
			{
				eUInt8 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8),
				eUInt16 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16),
				eUInt32 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32),
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