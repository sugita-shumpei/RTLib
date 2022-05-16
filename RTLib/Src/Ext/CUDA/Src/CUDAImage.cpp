#include <RTLib/Ext/CUDA/CUDAImage.h>
#include "CUDATypeConversions.h"
#include <iostream>

RTLib::Ext::CUDA::CUDAImage::CUDAImage(CUDAContext* ctx, const CUDAImageDesc& desc, CUarray cuArray, bool ownership)noexcept
{
	m_Context     = ctx;
	m_Width		  = desc.width;
	m_Height      = desc.height;
	m_Depth       = desc.depth;
	m_Layers      = desc.layers;
	m_Levels      = desc.levels;
	m_Format      = desc.format;
	m_ImageType   = desc.imageType;
	m_Channels = desc.channels;
	m_Flags       = desc.flags;
	m_ArrayMipmapped = nullptr;
	m_Arrays = { cuArray };
	m_Ownership = ownership;
}

RTLib::Ext::CUDA::CUDAImage::CUDAImage(CUDAContext* ctx, const CUDAImageDesc& desc, CUmipmappedArray cuArray, const std::vector<CUarray>& cuArrayRefs)noexcept
{

	m_Context = ctx;
	m_Width = desc.width;
	m_Height = desc.height;
	m_Depth = desc.depth;
	m_Layers = desc.layers;
	m_Levels = desc.levels;
	m_Format = desc.format;
	m_ImageType = desc.imageType;
	m_Channels = desc.channels;
	m_Flags = desc.flags;
	m_ArrayMipmapped = cuArray;
	m_Arrays = cuArrayRefs;
	m_Ownership = true;
}

auto RTLib::Ext::CUDA::CUDAImage::Allocate(CUDAContext* ctx, const CUDAImageDesc& desc)->CUDAImage*
{
	if (desc.width == 0) { return nullptr; }
	bool isMippedArray = desc.levels > 0;
	if (isMippedArray) {
		return AllocateMipmappedArray(ctx, desc);
	}
	else if ((desc.flags&CUDAImageFlagBitsCubemap)||(desc.layers>0)) {
		return AllocateArray3D(ctx, desc);
	}
	else {
		return AllocateArray(ctx, desc);
	}
}
void RTLib::Ext::CUDA::CUDAImage::Destroy() noexcept
{
	if (m_Ownership) {
		if (m_Arrays[0]) {
			auto res = cuArrayDestroy(m_Arrays[0]);
			if (res != CUDA_SUCCESS) {
				const char* errString = nullptr;
				(void)cuGetErrorString(res, &errString);
				std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			}
		}
		if (m_ArrayMipmapped) {
			auto res = cuMipmappedArrayDestroy(m_ArrayMipmapped);
			if (res != CUDA_SUCCESS) {
				const char* errString = nullptr;
				(void)cuGetErrorString(res, &errString);
				std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			}
			m_ArrayMipmapped = 0;
		}
	}
	m_Context        = nullptr;
	m_Width          = 0;
	m_Height         = 0;
	m_Depth			 = 0;
	m_Layers	     = 0;
	m_Levels         = 0;
	m_Format         = CUDAImageDataType::eUndefined;
	m_ImageType      = CUDAImageType::e1D;
	m_Channels       = 0;
	m_Flags          = 0;
	m_Arrays         = {};
	m_ArrayMipmapped = nullptr;
	m_Arrays = {};
	m_Ownership      = false;
}
auto RTLib::Ext::CUDA::CUDAImage::GetMipImage(unsigned int level) -> CUDAImage*
{
	auto cuArray = GetArrays(level);
	if (!cuArray) { return nullptr; }
	CUDAImageDesc desc = {};
	desc.imageType = this->m_ImageType;
	desc.width = m_Width;
	desc.height = m_Height;
	desc.depth = m_Depth;
	desc.layers = m_Layers;
	desc.levels = 0;
	desc.format = m_Format;
	desc.channels = m_Channels;
	desc.flags = m_Flags;
	return new CUDAImage(m_Context, desc, cuArray, false);
}
auto RTLib::Ext::CUDA::CUDAImage::GetMipWidth(unsigned int level) const noexcept -> size_t
{
	size_t width = m_Width;
	if (!width) { return 0; }
	for (unsigned int i = 0; i < level; ++i) {
		width = std::max<size_t>(1, width / 2);
	}
	return width;
}
auto RTLib::Ext::CUDA::CUDAImage::GetMipHeight(unsigned int level) const noexcept -> size_t
{
	size_t height = m_Height;
	if (!height) { return 0; }
	for (unsigned int i = 0; i < level; ++i) {
		height = std::max<size_t>(1, height / 2);
	}
	return height;
}
auto RTLib::Ext::CUDA::CUDAImage::GetMipDepth(unsigned int level) const noexcept -> size_t
{
	size_t depth = m_Depth;
	if (!depth) { return 0; }
	for (unsigned int i = 0; i < level; ++i) {
		depth = std::max<size_t>(1, depth / 2);
	}
	return depth;
}
RTLib::Ext::CUDA::CUDAImage::~CUDAImage() noexcept
{
}

auto RTLib::Ext::CUDA::CUDAImage::AllocateArray(CUDAContext* ctx, const CUDAImageDesc& desc) -> CUDAImage*
{
	CUDA_ARRAY_DESCRIPTOR arrDesc = {};
	switch (desc.imageType) {
	case CUDAImageType::e1D:
		if (desc.height> 0) { return nullptr; }
		if (desc.depth > 0) { return nullptr; }
		arrDesc.Width  = desc.width;
		arrDesc.Height = 0;
		break;
	case CUDAImageType::e2D:
		if (desc.depth > 0) { return nullptr; }
		arrDesc.Width  = desc.width;
		arrDesc.Height = desc.height;
		break;
	default:
		return nullptr;
	}
	arrDesc.Format      = GetCUDAImageDataTypeCUArrayFormat(desc.format);
	arrDesc.NumChannels = desc.channels;
	CUarray arr;
	{
		auto res = cuArrayCreate(&arr, &arrDesc);
		if (res != CUDA_SUCCESS) {
			const char* errString = nullptr;
			(void)cuGetErrorString(res, &errString);
			std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			return nullptr;
		}
	}
	return new CUDAImage(ctx,desc,nullptr,arr);
}

auto RTLib::Ext::CUDA::CUDAImage::AllocateArray3D(CUDAContext* ctx, const CUDAImageDesc& desc) -> CUDAImage*
{
	CUDA_ARRAY3D_DESCRIPTOR arrDesc = {};
	arrDesc.Flags = static_cast<uint32_t>(desc.flags);
	switch (desc.imageType) {
	case CUDAImageType::e1D:
		if (arrDesc.Flags & CUDAImageFlagBitsCubemap) {
			return nullptr;
		}
		arrDesc.Width  = desc.width;
		arrDesc.Height = 0;
		arrDesc.Depth  = desc.layers;
		break;
	case CUDAImageType::e2D:
		if (arrDesc.Flags & CUDAImageFlagBitsCubemap) {
			if (desc.depth > 0) { return nullptr; }
			if (desc.layers == 0) {
				if (desc.width != desc.height) { return nullptr; }
				arrDesc.Width  = desc.width;
				arrDesc.Height = desc.width;
				arrDesc.Depth  = 6;
			}
			else {
				arrDesc.Width  = desc.width;
				arrDesc.Height = desc.width;
				arrDesc.Depth  = 6 * desc.layers;
				arrDesc.Flags |= CUDA_ARRAY3D_LAYERED;
			}
		}
		else {
			if (desc.depth   > 0) { return nullptr; }
			if (desc.layers == 0) { return nullptr; }
			arrDesc.Width  = desc.width;
			arrDesc.Height = desc.height;
			arrDesc.Depth  = desc.layers;
			arrDesc.Flags |= CUDA_ARRAY3D_LAYERED;
		}
		break;
	case CUDAImageType::e3D:
		if (arrDesc.Flags & CUDAImageFlagBitsCubemap) {
			return nullptr;
		}
		if (desc.depth  == 0) { return nullptr; }
		if (desc.layers >  0) { return nullptr; }
		arrDesc.Width  = desc.width;
		arrDesc.Height = desc.height;
		arrDesc.Depth  = desc.depth;
		break;
	default:
		return nullptr;
	}
	arrDesc.Format      = GetCUDAImageDataTypeCUArrayFormat(desc.format);
	arrDesc.NumChannels = desc.channels;
	CUarray arr;
	{
		auto res = cuArray3DCreate(&arr, &arrDesc);
		if (res != CUDA_SUCCESS) {
			const char* errString = nullptr;
			(void)cuGetErrorString(res, &errString);
			std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			return nullptr;
		}
	}
	return new CUDAImage(ctx, desc, nullptr, arr);
}

auto RTLib::Ext::CUDA::CUDAImage::AllocateMipmappedArray(CUDAContext* ctx, const CUDAImageDesc& desc) -> CUDAImage*
{
	CUDA_ARRAY3D_DESCRIPTOR arrDesc = {};
	arrDesc.Flags = static_cast<uint32_t>(desc.flags);
	switch (desc.imageType) {
	case CUDAImageType::e1D:
		if (arrDesc.Flags & CUDAImageFlagBitsCubemap) {
			return nullptr;
		}
		arrDesc.Width  = desc.width;
		arrDesc.Height = 0;
		arrDesc.Depth  = desc.layers;
		break;
	case CUDAImageType::e2D:
		if (arrDesc.Flags & CUDAImageFlagBitsCubemap) {
			if (desc.layers == 0) {
				if (desc.width != desc.height) { return nullptr; }
				arrDesc.Width  = desc.width;
				arrDesc.Height = desc.width;
				arrDesc.Depth  = 6;
			}
			else {
				arrDesc.Width  = desc.width;
				arrDesc.Height = desc.width;
				arrDesc.Depth  = 6 * desc.layers;
				arrDesc.Flags |= CUDA_ARRAY3D_LAYERED;
			}
		}
		else {
			if (desc.layers == 0) {
				arrDesc.Width  = desc.width;
				arrDesc.Height = desc.height;
				arrDesc.Depth  = 0;
			}
			else {
				arrDesc.Width  = desc.width;
				arrDesc.Height = desc.height;
				arrDesc.Depth  = desc.layers;
				arrDesc.Flags |= CUDA_ARRAY3D_LAYERED;
			}
		}
		break;
	case CUDAImageType::e3D:
		if (arrDesc.Flags & CUDAImageFlagBitsCubemap) {
			return nullptr;
		}
		if (desc.depth == 0) { return nullptr; }
		if (desc.layers > 0) { return nullptr; }
		arrDesc.Width  = desc.width;
		arrDesc.Height = desc.height;
		arrDesc.Depth  = desc.depth;
		break;
	default:
		return nullptr;
	}
	arrDesc.Format      = GetCUDAImageDataTypeCUArrayFormat(desc.format);
	arrDesc.NumChannels = desc.channels;
	CUmipmappedArray arr;
	{
		auto res = cuMipmappedArrayCreate(&arr, &arrDesc,desc.levels);
		if (res != CUDA_SUCCESS) {
			const char* errString = nullptr;
			(void)cuGetErrorString(res, &errString);
			std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			return nullptr;
		}
	}
	std::vector<CUarray> arrRefs(desc.levels);
	for (auto i = 0; i < desc.levels;++i) {
		auto res = cuMipmappedArrayGetLevel(&arrRefs[i], arr,i);
		if (res != CUDA_SUCCESS) {
			const char* errString = nullptr;
			(void)cuGetErrorString(res, &errString);
			std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			return nullptr;
		}
	}
	return new CUDAImage(ctx, desc, arr, arrRefs);
}
