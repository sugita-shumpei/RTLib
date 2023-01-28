#include <RTLib/Core/TypeFormat.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include "CUDATypeConversions.h"
#include <iostream>
struct RTLib::Ext::CUDA::CUDAImage::Impl {
	Impl(CUDAContext* ctx, const CUDAImageCreateDesc& desc, CUarray cuArray, bool ownership)noexcept
	{
		m_Context         = ctx;
		m_Desc           = desc;
		m_ArrayMipmapped = nullptr;
		m_Arrays = { cuArray };
		m_Ownership = ownership;
	}
	Impl(CUDAContext* ctx, const CUDAImageCreateDesc& desc, CUmipmappedArray cuArray, const std::vector<CUarray>& cuArrayRefs, bool ownership)noexcept {

		m_Context = ctx;
		m_Desc = desc;
		m_ArrayMipmapped = cuArray;
		m_Arrays = cuArrayRefs;
		m_Ownership = ownership;
	}
	CUDAContext*            m_Context;
	CUmipmappedArray        m_ArrayMipmapped;
	std::vector<CUarray>    m_Arrays;
	CUDAImageCreateDesc     m_Desc;
	bool                    m_Ownership;
};
RTLib::Ext::CUDA::CUDAImage::CUDAImage(CUDAContext* ctx, const CUDAImageCreateDesc& desc, CUarray cuArray, bool ownership)noexcept :m_Impl{ new Impl(ctx,desc,cuArray,ownership) } {}


RTLib::Ext::CUDA::CUDAImage::CUDAImage(CUDAContext* ctx, const CUDAImageCreateDesc& desc, CUmipmappedArray cuArray, const std::vector<CUarray>& cuArrayRefs, bool ownership)noexcept
	:m_Impl{ new Impl(ctx,desc,cuArray,cuArrayRefs,ownership) } {}

auto RTLib::Ext::CUDA::CUDAImage::Allocate(CUDAContext* ctx, const CUDAImageCreateDesc& desc)->CUDAImage*
{
	if (desc.extent.width == 0) { return nullptr; }
	bool isMippedArray = desc.mipLevels > 0;
	if (isMippedArray) {
		return AllocateMipmappedArray(ctx, desc);
	}
	else if ((desc.flags&CUDAImageCreateFlagBitsCubemap)||(desc.arrayLayers>0)) {
		return AllocateArray3D(ctx, desc);
	}
	else {
		return AllocateArray(ctx, desc);
	}
}
void RTLib::Ext::CUDA::CUDAImage::Destroy() noexcept
{
	if (m_Impl->m_Ownership) {
		if (m_Impl->m_Arrays[0]) {
			auto res = cuArrayDestroy(m_Impl->m_Arrays[0]);
			if (res != CUDA_SUCCESS) {
				const char* errString = nullptr;
				(void)cuGetErrorString(res, &errString);
				std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			}
		}
		if (m_Impl->m_ArrayMipmapped) {
			auto res = cuMipmappedArrayDestroy(m_Impl->m_ArrayMipmapped);
			if (res != CUDA_SUCCESS) {
				const char* errString = nullptr;
				(void)cuGetErrorString(res, &errString);
				std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			}
			m_Impl->m_ArrayMipmapped = 0;
		}
	}
	m_Impl->m_Context = nullptr;
	m_Impl->m_Desc.extent.width = 0;
	m_Impl->m_Desc.extent.height = 0;
	m_Impl->m_Desc.extent.depth = 0;
	m_Impl->m_Desc.arrayLayers = 0;
	m_Impl->m_Desc.mipLevels = 0;
	m_Impl->m_Desc.format = CUDAImageFormat::eUndefined;
	m_Impl->m_Desc.imageType = CUDAImageType::e1D;
	m_Impl->m_Desc.flags = 0;
	m_Impl->m_ArrayMipmapped = nullptr;
	m_Impl->m_Arrays = {};
	m_Impl->m_Ownership = false;
}
auto RTLib::Ext::CUDA::CUDAImage::GetImageType() const noexcept -> CUDAImageType { return m_Impl->m_Desc.imageType; }
auto RTLib::Ext::CUDA::CUDAImage::GetWidth() const noexcept -> size_t { return m_Impl->m_Desc.extent.width; }
auto RTLib::Ext::CUDA::CUDAImage::GetHeight() const noexcept -> size_t { return m_Impl->m_Desc.extent.height; }
auto RTLib::Ext::CUDA::CUDAImage::GetDepth() const noexcept -> size_t { return m_Impl->m_Desc.extent.depth; }
auto RTLib::Ext::CUDA::CUDAImage::GetLevels() const noexcept -> size_t { return m_Impl->m_Desc.mipLevels; }
auto RTLib::Ext::CUDA::CUDAImage::GetLayers() const noexcept -> size_t { return m_Impl->m_Desc.arrayLayers; }
auto RTLib::Ext::CUDA::CUDAImage::GetFormat() const noexcept -> CUDAImageFormat { return m_Impl->m_Desc.format; }
auto RTLib::Ext::CUDA::CUDAImage::GetFlags() const noexcept -> unsigned int { return m_Impl->m_Desc.flags; }
auto RTLib::Ext::CUDA::CUDAImage::GetOwnership() const noexcept -> bool { return m_Impl->m_Ownership; }
auto RTLib::Ext::CUDA::CUDAImage::GetMipImage(unsigned int level) -> CUDAImage*
{
	auto cuArray = GetCUarrayWithLevel(level);
	if (!cuArray) { return nullptr; }
	CUDAImageCreateDesc desc = {};
	desc.imageType = this->m_Impl->m_Desc.imageType;
	desc.extent.width  = GetMipWidth(level);
	desc.extent.height = GetMipHeight(level);
	desc.extent.depth  = GetMipDepth(level);
	desc.arrayLayers = desc.arrayLayers;
	desc.mipLevels = 0;
	desc.format = desc.format;
	desc.flags = this->m_Impl->m_Desc.flags;
	return new CUDAImage(m_Impl->m_Context, desc, cuArray, false);
}
auto RTLib::Ext::CUDA::CUDAImage::GetMipWidth(unsigned int level) const noexcept -> size_t
{
	size_t width = m_Impl->m_Desc.extent.width;
	if (!width) { return 0; }
	for (unsigned int i = 0; i < level; ++i) {
		width = std::max<size_t>(1, width / 2);
	}
	return width;
}
auto RTLib::Ext::CUDA::CUDAImage::GetMipHeight(unsigned int level) const noexcept -> size_t
{
	size_t height = m_Impl->m_Desc.extent.height;
	if (!height) { return 0; }
	for (unsigned int i = 0; i < level; ++i) {
		height = std::max<size_t>(1, height / 2);
	}
	return height;
}
auto RTLib::Ext::CUDA::CUDAImage::GetMipDepth(unsigned int level) const noexcept -> size_t
{
	size_t depth = m_Impl->m_Desc.extent.depth;
	if (!depth) { return 0; }
	for (unsigned int i = 0; i < level; ++i) {
		depth = std::max<size_t>(1, depth / 2);
	}
	return depth;
}
RTLib::Ext::CUDA::CUDAImage::~CUDAImage() noexcept
{
}

auto RTLib::Ext::CUDA::CUDAImage::AllocateArray(CUDAContext* ctx, const CUDAImageCreateDesc& desc) -> CUDAImage*
{
	CUDA_ARRAY_DESCRIPTOR arrDesc = {};
	switch (desc.imageType) {
	case CUDAImageType::e1D:
		if (desc.extent.height> 0) { return nullptr; }
		if (desc.extent.depth > 0) { return nullptr; }
		arrDesc.Width  = desc.extent.width;
		arrDesc.Height = 0;
		break;
	case CUDAImageType::e2D:
		if (desc.extent.depth > 0) { return nullptr; }
		arrDesc.Width  = desc.extent.width;
		arrDesc.Height = desc.extent.height;
		break;
	default:
		return nullptr;
	}
	arrDesc.Format = GetCUDAImageDataTypeCUArrayFormat(CUDA::CUDAImageFormatUtils::GetChannelType(desc.format, 0));
	arrDesc.NumChannels = CUDA::CUDAImageFormatUtils::GetNumChannels(desc.format);
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

auto RTLib::Ext::CUDA::CUDAImage::AllocateArray3D(CUDAContext* ctx, const CUDAImageCreateDesc& desc) -> CUDAImage*
{
	CUDA_ARRAY3D_DESCRIPTOR arrDesc = {};
	arrDesc.Flags = static_cast<uint32_t>(desc.flags);
	switch (desc.imageType) {
	case CUDAImageType::e1D:
		if (arrDesc.Flags & CUDAImageCreateFlagBitsCubemap) {
			return nullptr;
		}
		arrDesc.Width  = desc.extent.width;
		arrDesc.Height = 0;
		arrDesc.Depth  = desc.arrayLayers;
		break;
	case CUDAImageType::e2D:
		if (arrDesc.Flags & CUDAImageCreateFlagBitsCubemap) {
			if (desc.extent.depth > 0) { return nullptr; }
			if (desc.arrayLayers == 0) {
				if (desc.extent.width != desc.extent.height) { return nullptr; }
				arrDesc.Width  = desc.extent.width;
				arrDesc.Height = desc.extent.width;
				arrDesc.Depth  = 6;
			}
			else {
				arrDesc.Width  = desc.extent.width;
				arrDesc.Height = desc.extent.width;
				arrDesc.Depth  = 6 * desc.arrayLayers;
				arrDesc.Flags |= CUDA_ARRAY3D_LAYERED;
			}
		}
		else {
			if (desc.extent.depth   > 0) { return nullptr; }
			if (desc.arrayLayers == 0) { return nullptr; }
			arrDesc.Width  = desc.extent.width;
			arrDesc.Height = desc.extent.height;
			arrDesc.Depth  = desc.arrayLayers;
			arrDesc.Flags |= CUDA_ARRAY3D_LAYERED;
		}
		break;
	case CUDAImageType::e3D:
		if (arrDesc.Flags & CUDAImageCreateFlagBitsCubemap) {
			return nullptr;
		}
		if (desc.extent.depth  == 0) { return nullptr; }
		if (desc.arrayLayers >  0) { return nullptr; }
		arrDesc.Width  = desc.extent.width;
		arrDesc.Height = desc.extent.height;
		arrDesc.Depth  = desc.extent.depth;
		break;
	default:
		return nullptr;
	}
	arrDesc.Format      = GetCUDAImageDataTypeCUArrayFormat(CUDA::CUDAImageFormatUtils::GetChannelType(desc.format, 0));
	arrDesc.NumChannels = CUDA::CUDAImageFormatUtils::GetNumChannels(desc.format);
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

auto RTLib::Ext::CUDA::CUDAImage::AllocateMipmappedArray(CUDAContext* ctx, const CUDAImageCreateDesc& desc) -> CUDAImage*
{
	CUDA_ARRAY3D_DESCRIPTOR arrDesc = {};
	arrDesc.Flags = static_cast<uint32_t>(desc.flags);
	switch (desc.imageType) {
	case CUDAImageType::e1D:
		if (arrDesc.Flags & CUDAImageCreateFlagBitsCubemap) {
			return nullptr;
		}
		arrDesc.Width  = desc.extent.width;
		arrDesc.Height = 0;
		arrDesc.Depth  = desc.arrayLayers;
		break;
	case CUDAImageType::e2D:
		if (arrDesc.Flags & CUDAImageCreateFlagBitsCubemap) {
			if (desc.arrayLayers == 0) {
				if (desc.extent.width != desc.extent.height) { return nullptr; }
				arrDesc.Width  = desc.extent.width;
				arrDesc.Height = desc.extent.width;
				arrDesc.Depth  = 6;
			}
			else {
				arrDesc.Width  = desc.extent.width;
				arrDesc.Height = desc.extent.width;
				arrDesc.Depth  = 6 * desc.arrayLayers;
				arrDesc.Flags |= CUDA_ARRAY3D_LAYERED;
			}
		}
		else {
			if (desc.arrayLayers == 0) {
				arrDesc.Width  = desc.extent.width;
				arrDesc.Height = desc.extent.height;
				arrDesc.Depth  = 0;
			}
			else {
				arrDesc.Width  = desc.extent.width;
				arrDesc.Height = desc.extent.height;
				arrDesc.Depth  = desc.arrayLayers;
				arrDesc.Flags |= CUDA_ARRAY3D_LAYERED;
			}
		}
		break;
	case CUDAImageType::e3D:
		if (arrDesc.Flags & CUDAImageCreateFlagBitsCubemap) {
			return nullptr;
		}
		if (desc.extent.depth == 0) { return nullptr; }
		if (desc.arrayLayers > 0) { return nullptr; }
		arrDesc.Width  = desc.extent.width;
		arrDesc.Height = desc.extent.height;
		arrDesc.Depth  = desc.extent.depth;
		break;
	default:
		return nullptr;
	}
	arrDesc.Format = GetCUDAImageDataTypeCUArrayFormat(CUDA::CUDAImageFormatUtils::GetChannelType(desc.format, 0));
	arrDesc.NumChannels = CUDA::CUDAImageFormatUtils::GetNumChannels(desc.format);
	CUmipmappedArray arr;
	{
		auto res = cuMipmappedArrayCreate(&arr, &arrDesc, desc.mipLevels);
		if (res != CUDA_SUCCESS) {
			const char* errString = nullptr;
			(void)cuGetErrorString(res, &errString);
			std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
			return nullptr;
		}
	}
	std::vector<CUarray> arrRefs(desc.mipLevels);
	for (auto i = 0; i < desc.mipLevels;++i) {
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

auto RTLib::Ext::CUDA::CUDAImage::GetCUarray() noexcept -> CUarray { return m_Impl->m_Arrays[0]; }

auto RTLib::Ext::CUDA::CUDAImage::GetCUarrayWithLevel(unsigned int level) noexcept -> CUarray {
	if (level >= m_Impl->m_Arrays.size()) { return nullptr; }
	return m_Impl->m_Arrays[level];
}

auto RTLib::Ext::CUDA::CUDAImage::GetCUmipmappedArray() noexcept -> CUmipmappedArray { return m_Impl->m_ArrayMipmapped; }
