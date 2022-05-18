#ifndef RTLIB_EXT_CUDA_CUDA_TYPE_FORMAT_H
#define RTLIB_EXT_CUDA_CUDA_TYPE_FORMAT_H
#include <RTLib/Core/TypeFormat.h>
namespace RTLib {
	namespace Ext {
		namespace CUDA {
			enum class CUDAImageDataType: uint64_t {
				eUndefined = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUndefined),
				eInt8      = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8),
				eInt16     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16),
				eInt32     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32),
				eUInt8     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8),
				eUInt16    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16),
				eUInt32    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32),
				eFloat16   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16),
				eFloat32   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32),
				eNV12      = 1<<9,
			};
			inline constexpr auto GetCUDAImageDataTypeSize(CUDAImageDataType dataType) -> uint32_t
			{
				return static_cast<uint32_t>(static_cast<uint64_t>(dataType) & 63);
			}
			static_assert(GetCUDAImageDataTypeSize(CUDAImageDataType::eFloat16) == 16);
			static_assert(GetCUDAImageDataTypeSize(CUDAImageDataType::eInt32) == 32);
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