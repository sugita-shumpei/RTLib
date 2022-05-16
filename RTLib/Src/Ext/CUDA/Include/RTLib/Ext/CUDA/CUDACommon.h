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
				case CU_AD_FORMAT_SIGNED_INT8: return CUDAImageDataType::eInt8;
				case CU_AD_FORMAT_SIGNED_INT16: return CUDAImageDataType::eInt16;
				case CU_AD_FORMAT_SIGNED_INT32: return CUDAImageDataType::eInt32;
				case CU_AD_FORMAT_UNSIGNED_INT8: return CUDAImageDataType::eUInt8;
				case CU_AD_FORMAT_UNSIGNED_INT16: return CUDAImageDataType::eUInt16;
				case CU_AD_FORMAT_UNSIGNED_INT32: return CUDAImageDataType::eUInt32;
				case CU_AD_FORMAT_HALF: return CUDAImageDataType::eFloat16;
				case CU_AD_FORMAT_FLOAT: return CUDAImageDataType::eFloat32;
				case CU_AD_FORMAT_NV12: return CUDAImageDataType::eNV12;
				default: return CUDAImageDataType::eNV12;
				}
			}
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
			inline constexpr auto GetCUDAResourceViewFormatCUResourceViewFormat(CUDAResourceViewFormat format)->CUresourceViewFormat
			{
				switch (format)
				{
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUndefined: return CU_RES_VIEW_FORMAT_NONE;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt8X1:return CU_RES_VIEW_FORMAT_SINT_1X8;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt8X2:return CU_RES_VIEW_FORMAT_SINT_2X8;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt8X4:return CU_RES_VIEW_FORMAT_SINT_4X8;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt8X1:return CU_RES_VIEW_FORMAT_UINT_1X8;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt8X2:return CU_RES_VIEW_FORMAT_UINT_2X8;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt8X4:return CU_RES_VIEW_FORMAT_UINT_4X8;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt16X1:return CU_RES_VIEW_FORMAT_SINT_1X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt16X2:return CU_RES_VIEW_FORMAT_SINT_2X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt16X4:return CU_RES_VIEW_FORMAT_SINT_4X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt16X1:return CU_RES_VIEW_FORMAT_UINT_1X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt16X2:return CU_RES_VIEW_FORMAT_UINT_2X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt16X4:return CU_RES_VIEW_FORMAT_UINT_4X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt32X1:return CU_RES_VIEW_FORMAT_SINT_1X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt32X2:return CU_RES_VIEW_FORMAT_SINT_2X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSInt32X4:return CU_RES_VIEW_FORMAT_SINT_4X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt32X1:return CU_RES_VIEW_FORMAT_UINT_1X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt32X2:return CU_RES_VIEW_FORMAT_UINT_2X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUInt32X4:return CU_RES_VIEW_FORMAT_UINT_4X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eFloat16X1:return CU_RES_VIEW_FORMAT_FLOAT_1X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eFloat16X2:return CU_RES_VIEW_FORMAT_FLOAT_2X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eFloat16X4:return CU_RES_VIEW_FORMAT_FLOAT_4X16;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eFloat32X1:return CU_RES_VIEW_FORMAT_FLOAT_1X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eFloat32X2:return CU_RES_VIEW_FORMAT_FLOAT_2X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eFloat32X4:return CU_RES_VIEW_FORMAT_FLOAT_4X32;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUnsignedBC1:return CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUnsignedBC2:return CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUnsignedBC3:return CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUnsignedBC4:return CU_RES_VIEW_FORMAT_UNSIGNED_BC4;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSignedBC4:return CU_RES_VIEW_FORMAT_SIGNED_BC4;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUnsignedBC5:return CU_RES_VIEW_FORMAT_UNSIGNED_BC5;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSignedBC5:return CU_RES_VIEW_FORMAT_SIGNED_BC5;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUnsignedBC6H:return CU_RES_VIEW_FORMAT_UNSIGNED_BC6H;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eSignedBC6H:return CU_RES_VIEW_FORMAT_SIGNED_BC6H;
					break;
				case RTLib::Ext::CUDA::CUDAResourceViewFormat::eUnsignedBC7:return CU_RES_VIEW_FORMAT_UNSIGNED_BC7;
					break;
				default: return CU_RES_VIEW_FORMAT_NONE;
					break;
				}
			}
			inline constexpr auto GetCUDAResourceViewFormatCUResourceViewFormat(CUresourceViewFormat format)->CUDAResourceViewFormat
			{
				switch (format)
				{
				case CU_RES_VIEW_FORMAT_NONE: return CUDAResourceViewFormat::eUndefined;
				case CU_RES_VIEW_FORMAT_SINT_1X8: return CUDAResourceViewFormat::eSInt8X1;
				case CU_RES_VIEW_FORMAT_SINT_2X8: return CUDAResourceViewFormat::eSInt8X2;
				case CU_RES_VIEW_FORMAT_SINT_4X8: return CUDAResourceViewFormat::eSInt8X4;
				case CU_RES_VIEW_FORMAT_UINT_1X8: return CUDAResourceViewFormat::eUInt8X1;
				case CU_RES_VIEW_FORMAT_UINT_2X8: return CUDAResourceViewFormat::eUInt8X2;
				case CU_RES_VIEW_FORMAT_UINT_4X8: return CUDAResourceViewFormat::eUInt8X4;
				case CU_RES_VIEW_FORMAT_SINT_1X16: return CUDAResourceViewFormat::eSInt16X1;
				case CU_RES_VIEW_FORMAT_SINT_2X16: return CUDAResourceViewFormat::eSInt16X2;
				case CU_RES_VIEW_FORMAT_SINT_4X16: return CUDAResourceViewFormat::eSInt16X4;
				case CU_RES_VIEW_FORMAT_UINT_1X16: return CUDAResourceViewFormat::eUInt16X1;
				case CU_RES_VIEW_FORMAT_UINT_2X16: return CUDAResourceViewFormat::eUInt16X2;
				case CU_RES_VIEW_FORMAT_UINT_4X16: return CUDAResourceViewFormat::eUInt16X4;
				case CU_RES_VIEW_FORMAT_SINT_1X32: return CUDAResourceViewFormat::eSInt32X1;
				case CU_RES_VIEW_FORMAT_SINT_2X32: return CUDAResourceViewFormat::eSInt32X2;
				case CU_RES_VIEW_FORMAT_SINT_4X32: return CUDAResourceViewFormat::eSInt32X4;
				case CU_RES_VIEW_FORMAT_UINT_1X32: return CUDAResourceViewFormat::eUInt32X1;
				case CU_RES_VIEW_FORMAT_UINT_2X32: return CUDAResourceViewFormat::eUInt32X2;
				case CU_RES_VIEW_FORMAT_UINT_4X32: return CUDAResourceViewFormat::eUInt32X4;
				case CU_RES_VIEW_FORMAT_FLOAT_1X16: return CUDAResourceViewFormat::eFloat16X1;
				case CU_RES_VIEW_FORMAT_FLOAT_2X16: return CUDAResourceViewFormat::eFloat16X2;
				case CU_RES_VIEW_FORMAT_FLOAT_4X16: return CUDAResourceViewFormat::eFloat16X4;
				case CU_RES_VIEW_FORMAT_FLOAT_1X32: return CUDAResourceViewFormat::eFloat32X1;
				case CU_RES_VIEW_FORMAT_FLOAT_2X32: return CUDAResourceViewFormat::eFloat32X2;
				case CU_RES_VIEW_FORMAT_FLOAT_4X32: return CUDAResourceViewFormat::eFloat32X4;
				case CU_RES_VIEW_FORMAT_UNSIGNED_BC1: return CUDAResourceViewFormat::eUnsignedBC1;
				case CU_RES_VIEW_FORMAT_UNSIGNED_BC2: return CUDAResourceViewFormat::eUnsignedBC2;
				case CU_RES_VIEW_FORMAT_UNSIGNED_BC3: return CUDAResourceViewFormat::eUnsignedBC3;
				case CU_RES_VIEW_FORMAT_UNSIGNED_BC4: return CUDAResourceViewFormat::eUnsignedBC4;
				case CU_RES_VIEW_FORMAT_SIGNED_BC4: return CUDAResourceViewFormat::eSignedBC4;
				case CU_RES_VIEW_FORMAT_UNSIGNED_BC5: return CUDAResourceViewFormat::eUnsignedBC5;
				case CU_RES_VIEW_FORMAT_SIGNED_BC5: return CUDAResourceViewFormat::eSignedBC5;
				case CU_RES_VIEW_FORMAT_UNSIGNED_BC6H: return CUDAResourceViewFormat::eUnsignedBC6H;
				case CU_RES_VIEW_FORMAT_SIGNED_BC6H: return CUDAResourceViewFormat::eSignedBC6H;
				case CU_RES_VIEW_FORMAT_UNSIGNED_BC7: return CUDAResourceViewFormat::eUnsignedBC7;
				default:return CUDAResourceViewFormat::eUndefined;
					break;
				}
			}
			enum class CUDAResourceType
			{
				eArray,
				eMipmappedArray,
				eLinear,
				ePitch2D
			};
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
			enum       CUDAImageFlagBits
			{
				CUDAImageFlagBitsDefault         = 0,
				CUDAImageFlagBitsSurfaceLDST     = CUDA_ARRAY3D_SURFACE_LDST,
				CUDAImageFlagBitsCubemap         = CUDA_ARRAY3D_CUBEMAP,
				CUDAImageFlagBitsDepthTexture    = CUDA_ARRAY3D_DEPTH_TEXTURE,
				CUDAImageFlagBitsTextureGather   = CUDA_ARRAY3D_TEXTURE_GATHER,
				CUDAImageFlagBitsColorAttachment = CUDA_ARRAY3D_COLOR_ATTACHMENT,
			};
			using      CUDAImageFlags = unsigned int;
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
			enum class CUDATextureAddressMode
			{
				eWarp   = static_cast<uint32_t>(Core::SamplerAddressMode::eRepeat),
				eClamp  = static_cast<uint32_t>(Core::SamplerAddressMode::eClampToEdge),
				eMirror = static_cast<uint32_t>(Core::SamplerAddressMode::eMirroredRepeat),
				eBorder = static_cast<uint32_t>(Core::SamplerAddressMode::eClampToBorder),
			};
			inline constexpr auto GetCUDATextureAddressModeCUAddressMode(CUDATextureAddressMode mode)->CUaddress_mode
			{
				switch (mode)
				{
				case RTLib::Ext::CUDA::CUDATextureAddressMode::eWarp: return CU_TR_ADDRESS_MODE_WRAP;
					break;
				case RTLib::Ext::CUDA::CUDATextureAddressMode::eClamp:return CU_TR_ADDRESS_MODE_CLAMP;
					break;
				case RTLib::Ext::CUDA::CUDATextureAddressMode::eMirror:return CU_TR_ADDRESS_MODE_MIRROR;
					break;
				case RTLib::Ext::CUDA::CUDATextureAddressMode::eBorder:return CU_TR_ADDRESS_MODE_BORDER;
					break;
				default: return CU_TR_ADDRESS_MODE_WRAP;
					break;
				}
			}
			inline constexpr auto GetCUAddressModeCUDATextureAddressMode(CUaddress_mode mode)->CUDATextureAddressMode
			{
				switch (mode)
				{
				case CU_TR_ADDRESS_MODE_WRAP: return CUDATextureAddressMode::eWarp;
					break;
				case CU_TR_ADDRESS_MODE_CLAMP:return CUDATextureAddressMode::eClamp;
					break;
				case CU_TR_ADDRESS_MODE_MIRROR:return CUDATextureAddressMode::eMirror;
					break;
				case CU_TR_ADDRESS_MODE_BORDER:return CUDATextureAddressMode::eBorder;
					break;
				default:
					break;
				}
			}
			enum class CUDATextureFilterMode
			{
				ePoint  = static_cast<uint32_t>(Core::FilterMode::eNearest),
				eLinear = static_cast<uint32_t>(Core::FilterMode::eLinear),
			};
			inline constexpr auto GetCUDATextureFilterModeCUFilterMode(CUDATextureFilterMode filter)->CUfilter_mode
			{
				switch (filter)
				{
				case RTLib::Ext::CUDA::CUDATextureFilterMode::ePoint: return CU_TR_FILTER_MODE_POINT;
					break;
				case RTLib::Ext::CUDA::CUDATextureFilterMode::eLinear:return CU_TR_FILTER_MODE_LINEAR;
					break;
				default: return CU_TR_FILTER_MODE_POINT;
					break;
				}
			}
			inline constexpr auto GetCUFilterModeCUDATextureFilterMode(CUfilter_mode filter)->CUDATextureFilterMode
			{
				switch (filter)
				{
				case CU_TR_FILTER_MODE_POINT: return RTLib::Ext::CUDA::CUDATextureFilterMode::ePoint;
					break;
				case CU_TR_FILTER_MODE_LINEAR: return RTLib::Ext::CUDA::CUDATextureFilterMode::eLinear;
					break;
				default:
					break;
				}
			}
			enum CUDATextureSamplerFlagBits
			{
				CUDATextureFlagBitsReadAsInteger                = CU_TRSF_READ_AS_INTEGER,
				CUDATextureFlagBitsNormalizedCordinates         = CU_TRSF_NORMALIZED_COORDINATES,
				CUDATextureFlagBitsDisableTrilinearOptimization = CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION,
			};
			using CUDATextureSamplerFlags = unsigned int;
			struct CUDATextureSamplerDesc
			{
				CUDATextureAddressMode  addressMode[3]      = {};
				float				    borderColor[4]      = {};
				CUDATextureFilterMode   filterMode          = CUDATextureFilterMode::ePoint;
				unsigned int	        maxAnisotropy       = 0;
				float				    maxMipmapLevelClamp = 1.0f;
				float				    minMipmapLevelClamp = 0.0f;
				CUDATextureFilterMode   mipmapFilterMode    = CUDATextureFilterMode::ePoint;
				float                   mipmapLevelBias     = 0.0f;
				CUDATextureSamplerFlags flags               = 0;
			};
			struct CUDATextureResourceViewDesc
			{
				CUDAResourceViewFormat format;
				size_t     		       width;
				size_t     		       height;
				size_t     		       depth;
				unsigned int           baseLayer;
				unsigned int           baseLevel;
				unsigned int           numLayers;
				unsigned int           numLevels;
			};
			class  CUDAImage;
			struct CUDATextureImageDesc
			{
				CUDAImage*                  image;
				CUDATextureResourceViewDesc view;
				CUDATextureSamplerDesc      sampler;
			};
		}
	}
}
#endif
