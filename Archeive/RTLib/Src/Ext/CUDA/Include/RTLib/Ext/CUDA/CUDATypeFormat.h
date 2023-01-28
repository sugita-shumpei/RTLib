#ifndef RTLIB_EXT_CUDA_CUDA_TYPE_FORMAT_H
#define RTLIB_EXT_CUDA_CUDA_TYPE_FORMAT_H
#include <RTLib/Core/TypeFormat.h>
namespace RTLib {
	namespace Ext {
		namespace CUDA {
			enum class CUDAImageDataType : uint64_t
			{
				eUndefined = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUndefined),
				eInt8      = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8),
				eInt16     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16),
				eInt32     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32),
				eUInt8     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8),
				eUInt16    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16),
				eUInt32    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32),
				eFloat16   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16),
				eFloat32   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32),
				eNV12      = 1 << 9,
			};
			enum class CUDAImageFormat   : uint64_t
			{
				eUndefined  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUndefined),
				eInt8X1     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X1),
				eInt8X2     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X2),
				eInt8X4     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X4),
				eInt16X1    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X1),
				eInt16X2    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X2),
				eInt16X4    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X4),
				eInt32X1    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X1),
				eInt32X2    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X2),
				eInt32X4    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X4),
				eUInt8X1    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X1),
				eUInt8X2    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X2),
				eUInt8X4    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X4),
				eUInt16X1   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X1),
				eUInt16X2   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X2),
				eUInt16X4   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X4),
				eUInt32X1   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X1),
				eUInt32X2   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X2),
				eUInt32X4   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X4),
				eFloat16X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X1),
				eFloat16X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X2),
				eFloat16X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X4),
				eFloat32X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X1),
				eFloat32X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X2),
				eFloat32X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X4),
				eNV12       = 1 << 9,
			};
			struct CUDAImageFormatUtils
			{
				static inline constexpr auto GetBitSize(CUDAImageFormat imageFormat)->uint32_t
				{
					if (imageFormat == CUDAImageFormat::eUndefined) { return 0; }
					if (imageFormat == CUDAImageFormat::eNV12     ) { return 8; }
					return Core::SizedTypeFlagsUtils::GetTypeBitSize(static_cast<Core::SizedTypeFlagBits>(imageFormat));
				}
				static inline constexpr bool HasChannelType(CUDAImageFormat imageFormat, unsigned int channel) {
					if (imageFormat == CUDAImageFormat::eUndefined) { return  channel == 0; }
					if (imageFormat == CUDAImageFormat::eNV12) { return channel==0; }
					return  Core::SizedTypeFlagsUtils::HasChannelType(static_cast<Core::SizedTypeFlagBits>(imageFormat), channel);
				}
				static inline constexpr auto GetChannelType(CUDAImageFormat imageFormat, unsigned int channel)->CUDAImageDataType
				{
					if (imageFormat == CUDAImageFormat::eUndefined) { return CUDAImageDataType::eUndefined; }
					if (imageFormat == CUDAImageFormat::eNV12) { return CUDAImageDataType::eNV12; }
					return static_cast<CUDAImageDataType>(
						Core::SizedTypeFlagsUtils::GetChannelType(static_cast<Core::SizedTypeFlagBits>(imageFormat), channel)
					);
				}
				static inline constexpr auto GetChannelTypeBitSize(CUDAImageFormat imageFormat, unsigned int channel)->unsigned int {
					auto channelTypeFlagBits = GetChannelType(imageFormat, channel);
					auto channelTypeSize = channelTypeFlagBits != CUDAImageDataType::eUndefined ? (static_cast<uint64_t>(channelTypeFlagBits) & 63) + 1 : 0;
					return channelTypeSize;
				}
				static inline constexpr auto GetNumChannels(CUDAImageFormat sizedType)->unsigned int {
					if (HasChannelType(sizedType, 3)) { return 4; }
					if (HasChannelType(sizedType, 2)) { return 3; }
					if (HasChannelType(sizedType, 1)) { return 2; }
					if (HasChannelType(sizedType, 0)) { return 1; }
					return 0;
				}
				
			};
			static_assert(CUDAImageFormatUtils::GetBitSize(CUDAImageFormat::eFloat16X1) == 16);
			static_assert(CUDAImageFormatUtils::GetBitSize(CUDAImageFormat::eInt32X2)   == 64);
			enum class CUDAResourceViewFormat : uint64_t
			{
				eUndefined = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUndefined),
				eSInt8X1   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X1),
				eSInt8X2   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X2),
				eSInt8X4   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X4),
				eUInt8X1   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X1),
				eUInt8X2   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X2),
				eUInt8X4   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X4),
				eSInt16X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X1),
				eSInt16X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X2),
				eSInt16X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X4),
				eUInt16X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X1),
				eUInt16X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X2),
				eUInt16X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X4),
				eSInt32X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X1),
				eSInt32X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X2),
				eSInt32X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X4),
				eUInt32X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X1),
				eUInt32X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X2),
				eUInt32X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X4),
				eFloat16X1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X1),
				eFloat16X2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X2),
				eFloat16X4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X4),
				eFloat32X1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X1),
				eFloat32X2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X2),
				eFloat32X4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X4),
				eUnsignedBC1  = (((uint64_t)1) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eUnsignedBC2  = (((uint64_t)2) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eUnsignedBC3  = (((uint64_t)3) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eUnsignedBC4  = (((uint64_t)4) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eSignedBC4    = (((uint64_t)4) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eInteger),
				eUnsignedBC5  = (((uint64_t)5) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eSignedBC5    = (((uint64_t)5) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eInteger),
				eUnsignedBC6H = (((uint64_t)6) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eSignedBC6H   = (((uint64_t)6) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eInteger),
				eUnsignedBC7  = (((uint64_t)7) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
			};
			
        }
    }
}

#endif