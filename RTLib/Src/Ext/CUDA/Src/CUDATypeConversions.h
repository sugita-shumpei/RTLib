#ifndef RTLIB_EXT_CUDA_CUDA_TYPE_CONVERSIONS_H
#define RTLIB_EXT_CUDA_CUDA_TYPE_CONVERSIONS_H
#include <RTLib/Ext/CUDA/CUDACommon.h>
namespace RTLib
{
	namespace Ext
	{
		namespace CUDA
		{
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
		}
	}
}
#endif
