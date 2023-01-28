#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLNatives.h>
#include <memory>
#include "GLTypeConversions.h"
struct RTLib::Ext::GL::GLTexture::Impl {
    std::unique_ptr<GLImage> image = nullptr;
};
auto RTLib::Ext::GL::GLTexture::Allocate(GLContext *context, const GLTextureCreateDesc &desc) -> GLTexture *
{
    auto texture = new GLTexture();
    texture->m_Impl->image = std::unique_ptr<GLImage>(GLImage::Allocate(context, desc.image));
    auto samplerDesc = desc.sampler;
    if (context->SupportVersion(4, 5)) {
        glTextureParameteri(texture->GetResId(), GL_TEXTURE_MAG_FILTER, GetGLMagFilterEnum(samplerDesc.magFilter));
        glTextureParameteri(texture->GetResId(), GL_TEXTURE_MIN_FILTER, GetGLMinFilterEnum(samplerDesc.minFilter, samplerDesc.mipmapMode));
        glTextureParameteri(texture->GetResId(), GL_TEXTURE_WRAP_S, GetGLAddressModeGLEnum(samplerDesc.addressModeU));
        glTextureParameteri(texture->GetResId(), GL_TEXTURE_WRAP_T, GetGLAddressModeGLEnum(samplerDesc.addressModeV));
        glTextureParameteri(texture->GetResId(), GL_TEXTURE_WRAP_R, GetGLAddressModeGLEnum(samplerDesc.addressModeW));
        glTextureParameterf(texture->GetResId(), GL_TEXTURE_LOD_BIAS, samplerDesc.mipLodBias);
        glTextureParameterf(texture->GetResId(), GL_TEXTURE_MIN_LOD, samplerDesc.minLod);
        glTextureParameterf(texture->GetResId(), GL_TEXTURE_MAX_LOD, samplerDesc.maxLod);
        glTextureParameterfv(texture->GetResId(), GL_TEXTURE_BORDER_COLOR, samplerDesc.borderColor);
        if (samplerDesc.anisotropyEnable)
        {
            glTextureParameterf(texture->GetResId(), GL_TEXTURE_MAX_ANISOTROPY, samplerDesc.maxAnisotropy);
        }
        if (samplerDesc.compareEnable)
        {
            glTextureParameteri(texture->GetResId(), GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
            glTextureParameteri(texture->GetResId(), GL_TEXTURE_COMPARE_FUNC, GetGLCompareOpGLEnum(samplerDesc.compareOp));
        }
    }
    else {
        context->SetTexture(0, texture);
        auto target = GetGLImageViewTypeGLenum(texture->m_Impl->image->GetViewType());
        bool useMap = desc.image.mipLevels > 1;
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GetGLMagFilterEnum(samplerDesc.magFilter));
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GetGLMinFilterEnum(samplerDesc.minFilter, samplerDesc.mipmapMode, useMap));
        glTexParameteri(target, GL_TEXTURE_WRAP_S, GetGLAddressModeGLEnum(samplerDesc.addressModeU));
        glTexParameteri(target, GL_TEXTURE_WRAP_T, GetGLAddressModeGLEnum(samplerDesc.addressModeV));
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GetGLAddressModeGLEnum(samplerDesc.addressModeW));
        glTexParameterf(target, GL_TEXTURE_LOD_BIAS, samplerDesc.mipLodBias);
        glTexParameterf(target, GL_TEXTURE_MIN_LOD, samplerDesc.minLod);
        glTexParameterf(target, GL_TEXTURE_MAX_LOD, samplerDesc.maxLod);
        glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, samplerDesc.borderColor);
        if (samplerDesc.anisotropyEnable)
        {
            glTexParameterf(target, GL_TEXTURE_MAX_ANISOTROPY, samplerDesc.maxAnisotropy);
        }
        if (samplerDesc.compareEnable)
        {
            glTexParameteri(target, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
            glTexParameteri(target, GL_TEXTURE_COMPARE_FUNC, GetGLCompareOpGLEnum(samplerDesc.compareOp));
        }
    }
    return texture;
}

RTLib::Ext::GL::GLTexture::~GLTexture() noexcept
{
    m_Impl.reset();
}

void RTLib::Ext::GL::GLTexture::Destroy() noexcept
{
    if (!m_Impl) { return; }
    if (!m_Impl->image) { return; }
    m_Impl->image->Destroy();
}

auto RTLib::Ext::GL::GLTexture::GetImage() noexcept -> GLImage*
{
    return m_Impl->image.get();
}

auto RTLib::Ext::GL::GLTexture::GetImage() const noexcept -> const GLImage*
{
    return m_Impl->image.get();
}

auto RTLib::Ext::GL::GLTexture::GetType() const noexcept -> GLImageViewType
{
    return GLImageViewType(m_Impl->image->GetViewType());
}

auto RTLib::Ext::GL::GLTexture::GetFormat() const noexcept -> GLFormat
{
    return GLFormat(m_Impl->image->GetFormat());
}

auto RTLib::Ext::GL::GLTexture::GetExtent() const noexcept -> GLExtent3D
{
    return GLExtent3D(m_Impl->image->GetExtent());
}

auto RTLib::Ext::GL::GLTexture::GetMipExtent(uint32_t level) const noexcept -> GLExtent3D
{
    return GLExtent3D(m_Impl->image->GetMipExtent(level));
}

auto RTLib::Ext::GL::GLTexture::GetMipLevels() const noexcept -> uint32_t
{
    return uint32_t(m_Impl->image->GetMipLevels());
}

auto RTLib::Ext::GL::GLTexture::GetArrayLayers() const noexcept -> uint32_t
{
    return uint32_t(m_Impl->image->GetArrayLayers());
}

auto RTLib::Ext::GL::GLTexture::GetFlags() const noexcept -> GLImageCreateFlags
{
    return GLImageCreateFlags(m_Impl->image->GetFlags());
}

RTLib::Ext::GL::GLTexture::GLTexture() noexcept : m_Impl{new Impl()}
{
}

auto RTLib::Ext::GL::GLTexture::GetResId() const noexcept -> GLuint
{
    return GLNatives::GetResId(m_Impl->image.get());
}
