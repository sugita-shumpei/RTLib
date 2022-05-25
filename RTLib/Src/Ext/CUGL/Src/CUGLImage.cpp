#include <RTLib/Ext/CUGL/CUGLImage.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <unordered_map>
struct  RTLib::Ext::CUGL::CUGLImage::Impl
{
    using UniqueCUDAImage = std::unique_ptr<CUDA::CUDAImage>;
    Impl(CUDA::CUDAContext* ctx, GL::GLImage* image)noexcept :ctxCUDA{ ctx }, imageGL{ image }, graphicsResource{ nullptr }, mipImageCU{ nullptr }, subImageCUs{}, isMapped{ false }{}
    CUDA::CUDAContext* ctxCUDA;
    GL::GLImage* imageGL;
    UniqueCUDAImage                              mipImageCU;
    std::unordered_map<uint64_t, UniqueCUDAImage> subImageCUs;
    CUgraphicsResource                           graphicsResource;
    bool                                         isMapped;
};

auto RTLib::Ext::CUGL::CUGLImage::New(CUDA::CUDAContext* ctx, GL::GLImage* imageGL, unsigned int flags) -> CUGLImage*
{
    auto image = new CUGLImage(ctx, imageGL);
    GL::GLImageViewType viewType = imageGL->GetViewType();
    GLenum target = 0;
    switch (viewType)
    {
    case RTLib::Ext::GL::GLImageViewType::e1D:
        target = GL_TEXTURE_1D;
        break;
    case RTLib::Ext::GL::GLImageViewType::e2D:
        target = GL_TEXTURE_2D;
        break;
    case RTLib::Ext::GL::GLImageViewType::e3D:
        target = GL_TEXTURE_3D;
        break;
    case RTLib::Ext::GL::GLImageViewType::eCubemap:
        target = GL_TEXTURE_CUBE_MAP;
        break;
    case RTLib::Ext::GL::GLImageViewType::e1DArray:
        target = GL_TEXTURE_1D_ARRAY;
        break;
    case RTLib::Ext::GL::GLImageViewType::e2DArray:
        target = GL_TEXTURE_2D_ARRAY;
        break;
    case RTLib::Ext::GL::GLImageViewType::eCubemapArray:
        target = GL_TEXTURE_CUBE_MAP_ARRAY;
        break;
    default:
        target = GL_TEXTURE_2D;
        break;
    }
    CUgraphicsResource resource;
    RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsGLRegisterImage(&resource, GL::GLNatives::GetResId(imageGL), target, flags));
    image->m_Impl->graphicsResource = resource;
    return image;
}

RTLib::Ext::CUGL::CUGLImage::~CUGLImage() noexcept
{
    m_Impl.reset();
}

void RTLib::Ext::CUGL::CUGLImage::Destroy() noexcept
{
    if (!m_Impl) { return; }
    if (m_Impl->graphicsResource) {
        try {
            RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsUnregisterResource(m_Impl->graphicsResource));
        }
        catch (CUDA::CUDAException& exception) {
            std::cerr << exception.what() << std::endl;
        }
    }
    m_Impl->ctxCUDA = nullptr;
    m_Impl->imageGL = nullptr;
    m_Impl->graphicsResource = nullptr;
}

auto RTLib::Ext::CUGL::CUGLImage::MapSubImage(uint32_t layer, uint32_t level, CUDA::CUDAStream* stream) -> CUDA::CUDAImage*
{
    if (m_Impl->mipImageCU) { return nullptr; }
    auto keyValue = ((static_cast<uint64_t>(layer) << 32) | static_cast<uint64_t>(level));
    {
        if (m_Impl->subImageCUs.count(keyValue) > 0) {
            auto subImage = m_Impl->subImageCUs.at(keyValue).get();
            if (subImage) {
                return subImage;
            }
        }
    }
    CUarray     arrayCU;
    auto imageCreateDesc = CUDA::CUDAImageCreateDesc();
    if (!m_Impl->isMapped) {
        RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsMapResources(1, &m_Impl->graphicsResource, CUDA::CUDANatives::GetCUstream(stream)));
    }
    RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsSubResourceGetMappedArray(&arrayCU, m_Impl->graphicsResource, layer, level));
    {
        CUDA_ARRAY3D_DESCRIPTOR cuDesc3d;
        RTLIB_EXT_CUDA_THROW_IF_FAILED(cuArray3DGetDescriptor(&cuDesc3d, arrayCU));
        imageCreateDesc.imageType     = m_Impl->imageGL->GetImageType();
        imageCreateDesc.extent.width  = cuDesc3d.Width;
        imageCreateDesc.extent.height = cuDesc3d.Height;
        imageCreateDesc.extent.depth  = cuDesc3d.Depth;
        imageCreateDesc.arrayLayers   = 0;
        imageCreateDesc.mipLevels     = 0;
        imageCreateDesc.flags         = cuDesc3d.Flags;
        imageCreateDesc.format        = CUDA::CUDAImageFormat::eUInt8X1;
        if (cuDesc3d.Format == CU_AD_FORMAT_SIGNED_INT8) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt8X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt8X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt8X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_UNSIGNED_INT8) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt8X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt8X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt8X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_SIGNED_INT16) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt16X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt16X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt16X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_UNSIGNED_INT16) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt16X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt16X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt16X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_SIGNED_INT32) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt32X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt32X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt32X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_UNSIGNED_INT32) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt32X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt32X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt32X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_HALF) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat16X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat16X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat16X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_FLOAT) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat32X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat32X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat32X4;
            }
        }
    }
    m_Impl->isMapped = true;
    auto subImage = CUDA::CUDANatives::GetCUDAImage(m_Impl->ctxCUDA, imageCreateDesc, arrayCU);
    m_Impl->subImageCUs[keyValue] = std::unique_ptr<CUDA::CUDAImage>(subImage);
    return subImage;
}

auto RTLib::Ext::CUGL::CUGLImage::MapMipImage(CUDA::CUDAStream* stream) -> CUDA::CUDAImage*
{
    if (m_Impl->mipImageCU) { return m_Impl->mipImageCU.get(); }
    CUmipmappedArray mipmappedArrayCU = nullptr;
    auto imageCreateDesc = CUDA::CUDAImageCreateDesc(); 
    if (!m_Impl->isMapped) {
        RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsMapResources(1, &m_Impl->graphicsResource, CUDA::CUDANatives::GetCUstream(stream)));
    }
    RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsResourceGetMappedMipmappedArray(&mipmappedArrayCU, m_Impl->graphicsResource));

    auto levels   = m_Impl->imageGL->GetMipLevels();
    auto arrayCUs = std::vector<CUarray>(levels,nullptr);
    for (int i = 0; i < levels; ++i) {
        RTLIB_EXT_CUDA_THROW_IF_FAILED(cuMipmappedArrayGetLevel(&arrayCUs[i], mipmappedArrayCU, i));
    }
    {
        CUDA_ARRAY3D_DESCRIPTOR cuDesc3d;
        RTLIB_EXT_CUDA_THROW_IF_FAILED(cuArray3DGetDescriptor(&cuDesc3d, arrayCUs[0]));
        imageCreateDesc.imageType = m_Impl->imageGL->GetImageType();
        imageCreateDesc.extent.width = cuDesc3d.Width;
        imageCreateDesc.extent.height = cuDesc3d.Height;
        imageCreateDesc.extent.depth = cuDesc3d.Depth;
        imageCreateDesc.arrayLayers = 0;
        imageCreateDesc.mipLevels = 0;
        imageCreateDesc.flags = cuDesc3d.Flags;
        imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt8X1;
        if (cuDesc3d.Format == CU_AD_FORMAT_SIGNED_INT8) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt8X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt8X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt8X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_UNSIGNED_INT8) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt8X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt8X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt8X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_SIGNED_INT16) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt16X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt16X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt16X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_UNSIGNED_INT16) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt16X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt16X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt16X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_SIGNED_INT32) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt32X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt32X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eInt32X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_UNSIGNED_INT32) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt32X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt32X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eUInt32X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_HALF) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat16X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat16X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat16X4;
            }
        }
        if (cuDesc3d.Format == CU_AD_FORMAT_FLOAT) {
            if (cuDesc3d.NumChannels == 1) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat32X1;
            }
            if (cuDesc3d.NumChannels == 2) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat32X2;
            }
            if (cuDesc3d.NumChannels == 4) {
                imageCreateDesc.format = CUDA::CUDAImageFormat::eFloat32X4;
            }
        }
    }
    auto mipImageCU = CUDA::CUDANatives::GetCUDAImage(m_Impl->ctxCUDA, imageCreateDesc, mipmappedArrayCU, arrayCUs);
    m_Impl->isMapped = true;
    m_Impl->mipImageCU = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(mipImageCU);
    return m_Impl->mipImageCU.get();
}

void RTLib::Ext::CUGL::CUGLImage::Unmap(CUDA::CUDAStream* stream)
{
    if (!m_Impl->isMapped) { return; }
    for (auto& [idx,subImageCU] : m_Impl->subImageCUs) {
        subImageCU->Destroy();
    }
    m_Impl->subImageCUs.clear();
    if (m_Impl->mipImageCU) {
        m_Impl->mipImageCU->Destroy();
        m_Impl->mipImageCU.reset();
    }
    RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsUnmapResources(1, &m_Impl->graphicsResource, CUDA::CUDANatives::GetCUstream(stream)));
    m_Impl->isMapped = false;
}

auto RTLib::Ext::CUGL::CUGLImage::GetContextCU() noexcept -> CUDA::CUDAContext*
{
    return m_Impl->ctxCUDA;
}

auto RTLib::Ext::CUGL::CUGLImage::GetContextCU() const noexcept -> const CUDA::CUDAContext*
{
    return m_Impl->ctxCUDA;
}

RTLib::Ext::CUGL::CUGLImage::CUGLImage(CUDA::CUDAContext* ctx, GL::GLImage* imageGL) noexcept:m_Impl{new Impl(ctx,imageGL)}
{
}
