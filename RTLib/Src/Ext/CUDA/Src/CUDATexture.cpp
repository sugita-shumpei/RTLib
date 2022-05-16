#include <RTLib/Ext/CUDA/CUDATexture.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include "CUDATypeConversions.h"
#include <iostream>
auto RTLib::Ext::CUDA::CUDATexture::Allocate(CUDAContext* context, const CUDATextureImageDesc& desc) -> CUDATexture*
{
    if (!context || !desc.image) { return nullptr; }
    auto numLevels  = desc.image->GetLevels();
    auto numLayers  = desc.image->GetLayers();
    auto hArray     = desc.image->GetArray();
    auto hMipmapped = desc.image->GetMipMappedArray();
    CUDA_RESOURCE_DESC resDesc = {};
    if (!hMipmapped) {
        resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
        resDesc.res.array.hArray = hArray;
    }
    else {
        resDesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        resDesc.res.mipmap.hMipmappedArray = hMipmapped;
    }
    resDesc.flags = 0;

    CUDA_RESOURCE_VIEW_DESC resViewDesc = {};
    resViewDesc.width                   = desc.view.width;
    resViewDesc.height                  = desc.view.height;
    resViewDesc.depth                   = desc.view.depth;
    resViewDesc.format                  = GetCUDAResourceViewFormatCUResourceViewFormat(desc.view.format);
    resViewDesc.firstLayer              = desc.view.baseLayer;
    resViewDesc.lastLayer               = std::max<size_t>(desc.view.baseLayer + desc.view.numLayers, 1) - 1;
    resViewDesc.firstMipmapLevel        = desc.view.baseLevel;
    resViewDesc.lastMipmapLevel         = std::max<size_t>(desc.view.baseLevel + desc.view.numLevels, 1) - 1;

    CUDA_TEXTURE_DESC texDesc   = {};
    texDesc.addressMode[0]      = GetCUDATextureAddressModeCUAddressMode(desc.sampler.addressMode[0]);
    texDesc.addressMode[1]      = GetCUDATextureAddressModeCUAddressMode(desc.sampler.addressMode[1]);
    texDesc.addressMode[2]      = GetCUDATextureAddressModeCUAddressMode(desc.sampler.addressMode[2]);
    texDesc.borderColor[0]      = desc.sampler.borderColor[0];
    texDesc.borderColor[1]      = desc.sampler.borderColor[1];
    texDesc.borderColor[2]      = desc.sampler.borderColor[2];
    texDesc.borderColor[3]      = desc.sampler.borderColor[3];
    texDesc.filterMode          = GetCUDATextureFilterModeCUFilterMode(desc.sampler.filterMode);
    texDesc.mipmapFilterMode    = GetCUDATextureFilterModeCUFilterMode(desc.sampler.mipmapFilterMode);
    texDesc.maxAnisotropy       = desc.sampler.maxAnisotropy;
    texDesc.maxMipmapLevelClamp = desc.sampler.maxMipmapLevelClamp;
    texDesc.minMipmapLevelClamp = desc.sampler.minMipmapLevelClamp;
    texDesc.mipmapLevelBias     = desc.sampler.mipmapLevelBias;
    texDesc.flags               = desc.sampler.flags;
    CUtexObject texObject;
    auto result = CUDA_SUCCESS;
    {
        //result = cuTexObjectCreate(&texObject, &resDesc, &texDesc, &resViewDesc);
        result = cuTexObjectCreate(&texObject, &resDesc, &texDesc, nullptr);
        if (result != CUDA_SUCCESS) {
            const char* errString = nullptr;
            (void)cuGetErrorString(result, &errString);
            std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
            return nullptr;
        }
    }
    return new CUDATexture(texObject);
}
RTLib::Ext::CUDA::CUDATexture::~CUDATexture() noexcept
{
}

void RTLib::Ext::CUDA::CUDATexture::Destroy()
{
    if (!m_TexObject) { return; }
    auto result = CUDA_SUCCESS;
    {
        result = cuTexObjectDestroy(m_TexObject);
        if (result != CUDA_SUCCESS) {
            const char* errString = nullptr;
            (void)cuGetErrorString(result, &errString);
            std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        }
        m_TexObject = 0;
    }
}

RTLib::Ext::CUDA::CUDATexture::CUDATexture(CUtexObject texObject) noexcept:m_TexObject{texObject}
{
}
