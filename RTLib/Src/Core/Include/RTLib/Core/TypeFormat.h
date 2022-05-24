#ifndef RTLIB_CORE_TYPE_FORMAT_H
#define RTLIB_CORE_TYPE_FORMAT_H
#include <cstdint>
#define RTLIB_CORE_CORE_MAKE_SIZED_TYPE(BASE_TYPE, SIZE) (((uint64_t)BASE_TYPE)|((uint64_t)SIZE-1))
#define RTLIB_CORE_CORE_FORMAT_DEF_1(VAL1) ((uint64_t)VAL1)
#define RTLIB_CORE_CORE_FORMAT_DEF_2(VAL1, VAL2) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9)
#define RTLIB_CORE_CORE_FORMAT_DEF_3(VAL1, VAL2, VAL3) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18)
#define RTLIB_CORE_CORE_FORMAT_DEF_4(VAL1, VAL2, VAL3, VAL4) ((uint64_t)VAL1) | (((uint64_t)VAL2) << 9) | (((uint64_t)VAL3) << 18) | (((uint64_t)VAL4) << 27)
namespace RTLib {
	namespace Core {
		struct BitMaskFlagsUtils
		{
			static constexpr auto Log2(uint64_t v)->unsigned int
			{
				uint32_t val = 0;
				v /= 2;
				while (v > 0) {
					v /= 2;
					val++;
				} ;
				return val;
			}
		};
		namespace {
			static_assert(BitMaskFlagsUtils::Log2(UINT8_MAX)  == 7);
			static_assert(BitMaskFlagsUtils::Log2(UINT16_MAX) == 15);
			static_assert(BitMaskFlagsUtils::Log2(UINT32_MAX) == 31);
			static_assert(BitMaskFlagsUtils::Log2(UINT64_MAX) == 63);
		}
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
			eInt8      = static_cast<uint64_t>(BaseTypeFlagBits::eInteger) |( 8 - 1),
			eInt16     = static_cast<uint64_t>(BaseTypeFlagBits::eInteger) |(16 - 1),
			eInt32     = static_cast<uint64_t>(BaseTypeFlagBits::eInteger) |(32 - 1),
			eUInt8     = static_cast<uint64_t>(BaseTypeFlagBits::eUnsigned)|( 8 - 1),
			eUInt16    = static_cast<uint64_t>(BaseTypeFlagBits::eUnsigned)|(16 - 1),
			eUInt32    = static_cast<uint64_t>(BaseTypeFlagBits::eUnsigned)|(32 - 1),
			eFloat16   = static_cast<uint64_t>(BaseTypeFlagBits::eFloat)   |(16 - 1),
			eFloat32   = static_cast<uint64_t>(BaseTypeFlagBits::eFloat)   |(32 - 1),
			eFloat64   = static_cast<uint64_t>(BaseTypeFlagBits::eFloat)   |(64 - 1),
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
			eFloat64X1 = RTLIB_CORE_CORE_FORMAT_DEF_1(eFloat64),
			eFloat64X2 = RTLIB_CORE_CORE_FORMAT_DEF_2(eFloat64, eFloat64),
			eFloat64X3 = RTLIB_CORE_CORE_FORMAT_DEF_3(eFloat64, eFloat64, eFloat64),
			eFloat64X4 = RTLIB_CORE_CORE_FORMAT_DEF_4(eFloat64, eFloat64, eFloat64, eFloat64),
			
		};
		struct SizedTypeFlagsUtils
		{
			static inline constexpr bool HasChannelType(SizedTypeFlagBits sizedType, unsigned int channel) {
				return  ((static_cast<uint64_t>(sizedType) >> (9 * channel)) & ((1 << 9) - 1)) != 0;
			}
			static inline constexpr auto GetChannelType(SizedTypeFlagBits sizedType, unsigned int channel)->SizedTypeFlagBits
			{
				return static_cast<SizedTypeFlagBits>( (static_cast<uint64_t>(sizedType) >> (9 * channel))&((1<<9)-1));
			}
			static inline constexpr bool IsChannelTypeFloat(SizedTypeFlagBits sizedType, unsigned int channel) {
				return static_cast<uint64_t>(GetChannelType(sizedType, channel)) & static_cast<uint64_t>(BaseTypeFlagBits::eFloat);
			}
			static inline constexpr bool IsChannelTypeSigned(SizedTypeFlagBits sizedType, unsigned int channel) {
				return static_cast<uint64_t>(GetChannelType(sizedType, channel)) & static_cast<uint64_t>(BaseTypeFlagBits::eInteger);
			}
			static inline constexpr bool IsChannelTypeUnsigned(SizedTypeFlagBits sizedType, unsigned int channel) {
				return static_cast<uint64_t>(GetChannelType(sizedType, channel)) & static_cast<uint64_t>(BaseTypeFlagBits::eUnsigned);
			}
			static inline constexpr auto GetChannelTypeBitSize(SizedTypeFlagBits sizedType, unsigned int channel)->unsigned int {
				auto channelTypeFlagBits = GetChannelType(sizedType, channel);
				auto channelTypeSize     = channelTypeFlagBits != SizedTypeFlagBits::eUndefined ? (static_cast<uint64_t>(channelTypeFlagBits) & 63) + 1 : 0;
				return channelTypeSize;
			}
			static inline constexpr auto GetNumChannels(SizedTypeFlagBits sizedType)->unsigned int {
				if (HasChannelType(sizedType, 3) ){ return 4; }
				if (HasChannelType(sizedType, 2)) { return 3; }
				if (HasChannelType(sizedType, 1)) { return 2; }
				if (HasChannelType(sizedType, 0)) { return 1; }
				return 0;
			}
			static inline constexpr auto GetTypeBitSize(SizedTypeFlagBits sizedType)->unsigned int
			{
				auto channel0TypeSize     = GetChannelTypeBitSize(sizedType, 0);
				if  (channel0TypeSize==0) { return 0; }
				auto channel1TypeSize     = GetChannelTypeBitSize(sizedType, 1);
				if  (channel1TypeSize==0) { return channel0TypeSize; }
				auto channel2TypeSize     = GetChannelTypeBitSize(sizedType, 2);
				if  (channel2TypeSize == 0) { return channel0TypeSize+ channel1TypeSize; }
				auto channel3TypeSize     = GetChannelTypeBitSize(sizedType, 3);
				if  (channel3TypeSize == 0) { return channel0TypeSize + channel1TypeSize + channel2TypeSize; }
				return channel0TypeSize + channel1TypeSize + channel2TypeSize + channel3TypeSize;
			}
		};
		namespace {
			static_assert(SizedTypeFlagsUtils::GetChannelType(SizedTypeFlagBits::eFloat64  , 0) == SizedTypeFlagBits::eFloat64);
			static_assert(SizedTypeFlagsUtils::GetChannelType(SizedTypeFlagBits::eFloat64  , 1) == SizedTypeFlagBits::eUndefined);
			static_assert(SizedTypeFlagsUtils::GetChannelType(SizedTypeFlagBits::eFloat64  , 2) == SizedTypeFlagBits::eUndefined);
			static_assert(SizedTypeFlagsUtils::GetChannelType(SizedTypeFlagBits::eFloat64  , 3) == SizedTypeFlagBits::eUndefined);
			static_assert(SizedTypeFlagsUtils::GetChannelType(SizedTypeFlagBits::eFloat64X4, 0) == SizedTypeFlagBits::eFloat64);
			static_assert(SizedTypeFlagsUtils::GetChannelType(SizedTypeFlagBits::eFloat64X4, 1) == SizedTypeFlagBits::eFloat64);
			static_assert(SizedTypeFlagsUtils::GetChannelType(SizedTypeFlagBits::eFloat64X4, 2) == SizedTypeFlagBits::eFloat64);
			static_assert(SizedTypeFlagsUtils::GetChannelType(SizedTypeFlagBits::eFloat64X4, 3) == SizedTypeFlagBits::eFloat64);
			static_assert(SizedTypeFlagsUtils::GetTypeBitSize(SizedTypeFlagBits::eInt8X1) ==  8);
			static_assert(SizedTypeFlagsUtils::GetTypeBitSize(SizedTypeFlagBits::eInt8X2) == 16);
			static_assert(SizedTypeFlagsUtils::GetTypeBitSize(SizedTypeFlagBits::eInt8X3) == 24);
			static_assert(SizedTypeFlagsUtils::GetTypeBitSize(SizedTypeFlagBits::eInt8X4) == 32);
			static_assert(SizedTypeFlagsUtils::GetChannelTypeBitSize(SizedTypeFlagBits::eFloat64, 0) == 64);
			static_assert(SizedTypeFlagsUtils::GetChannelTypeBitSize(SizedTypeFlagBits::eFloat64, 1) == 0);
			static_assert(SizedTypeFlagsUtils::GetChannelTypeBitSize(SizedTypeFlagBits::eFloat64, 2) == 0);
			static_assert(SizedTypeFlagsUtils::GetChannelTypeBitSize(SizedTypeFlagBits::eFloat64, 3) == 0);
			static_assert(SizedTypeFlagsUtils::GetTypeBitSize(SizedTypeFlagBits::eFloat64X1) ==  64);
			static_assert(SizedTypeFlagsUtils::GetTypeBitSize(SizedTypeFlagBits::eFloat64X2) == 128);
			static_assert(SizedTypeFlagsUtils::GetTypeBitSize(SizedTypeFlagBits::eFloat64X3) == 192);
			static_assert(SizedTypeFlagsUtils::GetTypeBitSize(SizedTypeFlagBits::eFloat64X4) == 256);
			static_assert(SizedTypeFlagsUtils::IsChannelTypeFloat(SizedTypeFlagBits::eFloat64X1, 0));
			static_assert(SizedTypeFlagsUtils::IsChannelTypeFloat(SizedTypeFlagBits::eFloat64X2, 1));
			static_assert(SizedTypeFlagsUtils::IsChannelTypeFloat(SizedTypeFlagBits::eFloat64X3, 2));
			static_assert(SizedTypeFlagsUtils::IsChannelTypeFloat(SizedTypeFlagBits::eFloat64X4, 3));
			static_assert(SizedTypeFlagsUtils::GetNumChannels(SizedTypeFlagBits::eFloat64X1) == 1);
			static_assert(SizedTypeFlagsUtils::GetNumChannels(SizedTypeFlagBits::eFloat64X2) == 2);
			static_assert(SizedTypeFlagsUtils::GetNumChannels(SizedTypeFlagBits::eFloat64X3) == 3);
			static_assert(SizedTypeFlagsUtils::GetNumChannels(SizedTypeFlagBits::eFloat64X4) == 4);
			static_assert(SizedTypeFlagsUtils::IsChannelTypeSigned(SizedTypeFlagBits::eInt32X1, 0));
			static_assert(SizedTypeFlagsUtils::IsChannelTypeSigned(SizedTypeFlagBits::eInt32X2, 1));
			static_assert(SizedTypeFlagsUtils::IsChannelTypeSigned(SizedTypeFlagBits::eInt32X3, 2));
			static_assert(SizedTypeFlagsUtils::IsChannelTypeSigned(SizedTypeFlagBits::eInt32X4, 3));
		}
		enum class AttachmentCompponent :uint64_t
		{
			eRed         = ((uint64_t)1) << 36,
			eGreen       = ((uint64_t)1) << 37,
			eBlue        = ((uint64_t)1) << 38,
			eAlpha       = ((uint64_t)1) << 39,
			eDepth       = ((uint64_t)1) << 40,
			eStencil     = ((uint64_t)1) << 41,
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
    }
}
#undef RTLIB_CORE_CORE_FORMAT_DEF_1
#undef RTLIB_CORE_CORE_FORMAT_DEF_2
#undef RTLIB_CORE_CORE_FORMAT_DEF_3
#undef RTLIB_CORE_CORE_FORMAT_DEF_4
#endif