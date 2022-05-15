#ifndef RTLIB_EXT_CUDA_CUDA_COMMON_H
#define RTLIB_EXT_CUDA_CUDA_COMMON_H
#include <RTLib/Core/Common.h>
#include <cuda.h>
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
			inline constexpr auto GetCUDAImageDataTypeCUArrayFormat(CUDAImageDataType dataType)->CUarray_format
			{
				switch (dataType)
				{
				case RTLib::Ext::CUDA::CUDAImageDataType::eInt8:
					return CU_AD_FORMAT_SIGNED_INT8;
					break;
				case RTLib::Ext::CUDA::CUDAImageDataType::eInt16:
					return CU_AD_FORMAT_SIGNED_INT16;
					break;
				case RTLib::Ext::CUDA::CUDAImageDataType::eInt32:
					return CU_AD_FORMAT_SIGNED_INT8;
					break;
				case RTLib::Ext::CUDA::CUDAImageDataType::eUInt8:
					return CU_AD_FORMAT_UNSIGNED_INT8;
					break;
				case RTLib::Ext::CUDA::CUDAImageDataType::eUInt16:
					return CU_AD_FORMAT_UNSIGNED_INT16;
					break;
				case RTLib::Ext::CUDA::CUDAImageDataType::eUInt32:
					return CU_AD_FORMAT_UNSIGNED_INT32;
					break;
				case RTLib::Ext::CUDA::CUDAImageDataType::eFloat16:
					return CU_AD_FORMAT_HALF;
					break;
				case RTLib::Ext::CUDA::CUDAImageDataType::eFloat32:
					return CU_AD_FORMAT_FLOAT;
					break;
				case RTLib::Ext::CUDA::CUDAImageDataType::eNV12:
					return CU_AD_FORMAT_NV12;
				default:
					return CU_AD_FORMAT_FLOAT;
					break;
				}
			}
			inline constexpr auto GetCUArrayFormatCUDAImageDataType(CUarray_format format)->CUDAImageDataType
			{
				switch (format)
				{
				case CU_AD_FORMAT_SIGNED_INT8   : return CUDAImageDataType::eInt8;
				case CU_AD_FORMAT_SIGNED_INT16  : return CUDAImageDataType::eInt16;
				case CU_AD_FORMAT_SIGNED_INT32  : return CUDAImageDataType::eInt32;
				case CU_AD_FORMAT_UNSIGNED_INT8 : return CUDAImageDataType::eUInt8;
				case CU_AD_FORMAT_UNSIGNED_INT16: return CUDAImageDataType::eUInt16;
				case CU_AD_FORMAT_UNSIGNED_INT32: return CUDAImageDataType::eUInt32;
				case CU_AD_FORMAT_HALF          : return CUDAImageDataType::eFloat16;
				case CU_AD_FORMAT_FLOAT         : return CUDAImageDataType::eFloat32;
				case CU_AD_FORMAT_NV12          : return CUDAImageDataType::eNV12;
				default: return CUDAImageDataType::eNV12;
				}
			}
			inline constexpr auto GetCUDAImageDataTypeSize(CUDAImageDataType dataType) -> uint32_t
			{
				return static_cast<uint32_t>(static_cast<uint64_t>(dataType) & 63);
			}
			static_assert(GetCUDAImageDataTypeSize(CUDAImageDataType::eFloat16) == 16);
			static_assert(GetCUDAImageDataTypeSize(CUDAImageDataType::eInt32)   == 32);
			

			using CUDAImageSubresourceLayers = Core::ImageSubresourceLayers;
			using CUDABufferCopy       = Core::BufferCopy;
			using CUDABufferImageCopy  = Core::BufferImageCopy;
			using CUDAMemoryBufferCopy = Core::MemoryBufferCopy;
			using CUDABufferMemoryCopy = Core::BufferMemoryCopy;
			using CUDAImageMemoryCopy  = Core::ImageMemoryCopy;
			using CUDAMemoryImageCopy  = Core::MemoryImageCopy;
			using CUDAImageCopy        = Core::ImageCopy;

			enum class CUDAMemoryFlags {
				eDefault,
				ePageLocked   ,
			};
			struct     CUDABufferDesc {
				CUDAMemoryFlags flags = CUDAMemoryFlags::eDefault;
				size_t sizeInBytes = 0;
			};
			enum class CUDAImageType {
				e1D,
				e2D,
				e3D
			};
			enum CUDAImageFlagBits
			{
				CUDAImageFlagBitsDefault         = 0,
				CUDAImageFlagBitsSurfaceLDST     = CUDA_ARRAY3D_SURFACE_LDST,
				CUDAImageFlagBitsCubemap         = CUDA_ARRAY3D_CUBEMAP,
				CUDAImageFlagBitsDepthTexture    = CUDA_ARRAY3D_DEPTH_TEXTURE,
				CUDAImageFlagBitsTextureGather   = CUDA_ARRAY3D_TEXTURE_GATHER,
				CUDAImageFlagBitsColorAttachment = CUDA_ARRAY3D_COLOR_ATTACHMENT,
			};
			using CUDAImageFlags = unsigned int;
			struct     CUDAImageDesc {
				CUDAImageType     imageType   = CUDAImageType::e1D;
				unsigned int      width       = 1;
				unsigned int      height      = 0;
				unsigned int      depth       = 0;
				unsigned int      levels      = 0;
				unsigned int      layers      = 0;
				CUDAImageDataType format      = CUDAImageDataType::eUndefined;
				unsigned int      channels = 1;
				CUDAImageFlags    flags       = CUDAImageFlagBitsDefault;
			};
		}
	}
}
#endif
