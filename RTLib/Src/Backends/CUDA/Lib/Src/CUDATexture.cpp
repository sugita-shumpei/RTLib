#include <RTLib/Backends/CUDA/CUDATexture.h>
#include <RTLib/Backends/CUDA/CUDAMipmappedArray.h>
#include <RTLib/Backends/CUDA/CUDAArray.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include "CUDAInternals.h"
#include <vector>
#include <memory>

struct RTLib::Backends::Cuda::Texture::Impl
{
	Impl(const Array* arr, const TextureDesc& texDesc_) noexcept :
		texObject{ 0 },
		resType{
		TextureResourceType::eArray
	}, pOwner{ arr }, texDesc{ texDesc_ }
	{
		CUDA_RESOURCE_DESC resDesc = {};
		resDesc.flags = 0;
		resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
		resDesc.res.array.hArray = Internals::GetCUarray(arr);
		CUDA_TEXTURE_DESC tmpDesc = {};
		Internals::SetCudaTextureDesc(tmpDesc, texDesc);
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuTexObjectCreate(&texObject, &resDesc, &tmpDesc, nullptr));

	}
	Impl(const MipmappedArray* mipmappedArray, const TextureDesc& texDesc_) noexcept:
		texObject{ 0 },
		resType {
		TextureResourceType::eMipmappedArray
	}, pOwner{ mipmappedArray }, texDesc{ texDesc_ }
	{
		CUDA_RESOURCE_DESC resDesc = {};
		resDesc.flags = 0;
		resDesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
		resDesc.res.mipmap.hMipmappedArray = Internals::GetCUmipmappedArray(mipmappedArray);
		CUDA_TEXTURE_DESC tmpDesc = {};
		Internals::SetCudaTextureDesc(tmpDesc, texDesc);
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuTexObjectCreate(&texObject, &resDesc, &tmpDesc, nullptr));

	}
	~Impl()noexcept {
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuTexObjectDestroy(texObject));
	}
	CUtexObject         texObject;
	TextureDesc         texDesc;
	TextureResourceType resType;
	const void*         pOwner;
};
RTLib::Backends::Cuda::Texture::Texture(const MipmappedArray *mipmappedArray, const TextureDesc &desc) noexcept
	: m_Impl(new Impl(mipmappedArray,desc))
{
	
}

RTLib::Backends::Cuda::Texture::Texture(const Array *array_, const TextureDesc &desc) noexcept
	: m_Impl(new Impl(array_,desc))
{
	
}

RTLib::Backends::Cuda::Texture::~Texture() noexcept
{
	m_Impl.reset();
}

auto RTLib::Backends::Cuda::Texture::GetHandle() const noexcept -> void * {
	return reinterpret_cast<void*>(static_cast<uintptr_t>(m_Impl->texObject));
}

auto RTLib::Backends::Cuda::Texture::GetResourceType() const noexcept -> TextureResourceType {
	return m_Impl->resType;
}

auto RTLib::Backends::Cuda::Texture::GetMipmappedArray() const noexcept -> const MipmappedArray * {
	if (m_Impl->resType == TextureResourceType::eMipmappedArray) { return static_cast<const MipmappedArray*>(m_Impl->pOwner); }
	else {
		return nullptr;
	}
}

auto RTLib::Backends::Cuda::Texture::GetArray() const noexcept -> const Array* {

	if (m_Impl->resType == TextureResourceType::eArray) { return static_cast<const Array*>(m_Impl->pOwner); }
	else {
		return nullptr;
	}
}
auto RTLib::Backends::Cuda::Texture::GetDesc() const noexcept -> const TextureDesc&
{
	// TODO: return ステートメントをここに挿入します
	return m_Impl->texDesc;
}