#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <cassert>
#include "GLTypeConversions.h"
struct RTLib::Ext::GL::GLImage::Impl
{
    Impl(GLContext* ctx, const GLImageCreateDesc& desc, bool hasOwner = true)noexcept :context{ctx}
    {
        imageType   = desc.imageType;
        switch (imageType)
        {
        case RTLib::Ext::GL::GLImageType::e1D:
            if (desc.arrayLayers > 0) {
                viewType      = GL::GLImageViewType::e1DArray;
                extent.width  = desc.extent.width;
                extent.height = 0;
                extent.depth  = 0;
                arrayLayers   = desc.arrayLayers;
            }
            else {
                viewType      = GL::GLImageViewType::e1D;
                extent.width  = desc.extent.width;
                extent.height = 0;
                extent.depth  = 0;
                arrayLayers   = 0;
            } 
            break;
        case RTLib::Ext::GL::GLImageType::e2D:
            if (desc.flags & GLImageCreateFlagsCubemap) {
                if (desc.arrayLayers > 0) {
                    viewType      = GL::GLImageViewType::eCubemapArray;
                    extent.width  = desc.extent.width;
                    extent.height = desc.extent.height;
                    extent.depth  = 0;
                    arrayLayers   = desc.arrayLayers;
                }
                else {
                    viewType      = GL::GLImageViewType::eCubemap;
                    extent.width  = desc.extent.width;
                    extent.height = desc.extent.height;
                    extent.depth  = 0;
                    arrayLayers   = 0;
                }
            }
            else {
                if (desc.arrayLayers > 0) {
                    viewType      = GL::GLImageViewType::e2DArray;
                    extent.width  = desc.extent.width;
                    extent.height = desc.extent.height;
                    extent.depth  = 0;
                    arrayLayers   = desc.arrayLayers;
                }
                else {
                    viewType      = GL::GLImageViewType::e2D;
                    extent.width  = desc.extent.width;
                    extent.height = desc.extent.height;
                    extent.depth  = 0;
                    arrayLayers   = 0;
                }
            }
            break;
        case RTLib::Ext::GL::GLImageType::e3D:
            viewType      = GL::GLImageViewType::e3D;
            extent.width  = desc.extent.width;
            extent.height = desc.extent.height;
            extent.depth  = desc.extent.depth;
            arrayLayers   = 0;
            break;
        default:
            viewType      = GL::GLImageViewType::e1D;
            extent.width  = 0;
            extent.height = 0;
            extent.depth  = 0;
            arrayLayers   = 0;
            break;
        }
        flags       = desc.flags;
        format      = desc.format;
        mipLevels   = desc.mipLevels;
        hasOwnership = hasOwner;
        
    }
    GLContext*         context;
    GLImageViewType    viewType;
    GLuint             resId;
    GLImageCreateFlags flags;
    GLImageType        imageType;
    GLFormat           format;
    GLExtent3D         extent;
    uint32_t           mipLevels;
    uint32_t           arrayLayers;
    bool               hasOwnership;
};
auto RTLib::Ext::GL::GLImage::Allocate(GLContext* context, const GLImageCreateDesc& desc) -> GLImage*
{
    GLenum target;
    GLImageType imageType  = desc.imageType;
    GLsizei width          = 0;
    GLsizei height         = 0;
    GLsizei depth          = 0;
    GLsizei levels         = std::max<GLsizei>(desc.mipLevels, 1);
    GLuint  resIdx         = 0;
    GLenum  internalFormat = GetGLFormatGLenum(desc.format);
    GLenum  baseFormat     = GetGLBaseFormatGLenum(GLFormatUtils::GetBaseFormat(desc.format));
    auto    unpackType     = GetGLFormatGLUnpackEnum(desc.format);
    bool    isCubemap      = false;
    bool    isCompressed   = GLFormatUtils::IsComressed(desc.format);
    switch (imageType)
    {
    case RTLib::Ext::GL::GLImageType::e1D:
        if (desc.arrayLayers >= 1)
        {
            target = GL_TEXTURE_1D_ARRAY;
            width = desc.extent.width;
            height = desc.arrayLayers;
        }
        else
        {
            target = GL_TEXTURE_1D;
            width = desc.extent.width;
        }
        break;
    case RTLib::Ext::GL::GLImageType::e2D:
        if ((desc.flags & GLImageCreateFlagsCubemap) == GLImageCreateFlagsCubemap)
        {
            isCubemap = true;
            if (desc.arrayLayers >= 1)
            {
                target = GL_TEXTURE_CUBE_MAP_ARRAY;
                width = desc.extent.width;
                height = desc.extent.height;
                depth = desc.arrayLayers;
            }
            else
            {
                target = GL_TEXTURE_CUBE_MAP;
                width = desc.extent.width;
                height = desc.extent.height;
            }
        }
        else
        {
            if (desc.arrayLayers >= 1)
            {
                target = GL_TEXTURE_2D_ARRAY;
                width = desc.extent.width;
                height = desc.extent.height;
                depth = desc.arrayLayers;
            }
            else
            {
                target = GL_TEXTURE_2D;
                width = desc.extent.width;
                height = desc.extent.height;
            }
        }
        break;
    case RTLib::Ext::GL::GLImageType::e3D:
        target = GL_TEXTURE_3D;
        width = desc.extent.width;
        height = desc.extent.height;
        depth = desc.extent.depth;
        break;
    default:
        break;
    }
    glGenTextures(1, &resIdx);
    glBindTexture(target, resIdx);
    if (context->SupportVersion(4, 2))
    {
        if (depth > 0)
        {
            glTexStorage3D(target, levels, internalFormat, width, height, depth);
        }
        else if (height > 0)
        {
            glTexStorage2D(target, levels, internalFormat, width, height);
        }
        else
        {
            glTexStorage1D(target, levels, internalFormat, width);
        }
    }
    else
    {
        GLsizei mipWidth = width;
        GLsizei mipHeight = height;
        GLsizei mipDepth = depth;
        if (depth > 0)
        {
            if (target == GL_TEXTURE_CUBE_MAP_ARRAY)
            {
                for (int i = 0; i < levels; ++i)
                {
                    glTexImage3D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, i, internalFormat, mipWidth, mipHeight, depth, 0, baseFormat, unpackType, nullptr);
                    glTexImage3D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, i, internalFormat, mipWidth, mipHeight, depth, 0, baseFormat, unpackType, nullptr);
                    glTexImage3D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, i, internalFormat, mipWidth, mipHeight, depth, 0, baseFormat, unpackType, nullptr);
                    glTexImage3D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, i, internalFormat, mipWidth, mipHeight, depth, 0, baseFormat, unpackType, nullptr);
                    glTexImage3D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, i, internalFormat, mipWidth, mipHeight, depth, 0, baseFormat, unpackType, nullptr);
                    glTexImage3D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, i, internalFormat, mipWidth, mipHeight, depth, 0, baseFormat, unpackType, nullptr);
                    mipWidth = std::max<GLsizei>(1, mipWidth / 2);
                    mipHeight = std::max<GLsizei>(1, mipHeight / 2);
                }
            }
            if (target == GL_TEXTURE_3D)
            {
                for (int i = 0; i < levels; ++i)
                {
                    glTexImage3D(GL_TEXTURE_3D, i, internalFormat, mipWidth, mipHeight, mipDepth, 0, baseFormat, unpackType, nullptr);
                    mipWidth = std::max<GLsizei>(1, mipWidth / 2);
                    mipHeight = std::max<GLsizei>(1, mipHeight / 2);
                    mipDepth = std::max<GLsizei>(1, mipDepth / 2);
                }
            }
            if (target == GL_TEXTURE_2D_ARRAY)
            {
                for (int i = 0; i < levels; ++i)
                {
                    glTexImage3D(GL_TEXTURE_2D_ARRAY, i, internalFormat, mipWidth, mipHeight, depth, 0, baseFormat, unpackType, nullptr);
                    mipWidth = std::max<GLsizei>(1, mipWidth / 2);
                    mipHeight = std::max<GLsizei>(1, mipHeight / 2);
                    mipDepth = std::max<GLsizei>(1, mipDepth / 2);
                }
            }
        }
        else if (height > 0)
        {
            if (target == GL_TEXTURE_CUBE_MAP)
            {
                for (int i = 0; i < levels; ++i)
                {
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, i, internalFormat, mipWidth, mipHeight, 0, baseFormat, unpackType, nullptr);
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, i, internalFormat, mipWidth, mipHeight, 0, baseFormat, unpackType, nullptr);
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, i, internalFormat, mipWidth, mipHeight, 0, baseFormat, unpackType, nullptr);
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, i, internalFormat, mipWidth, mipHeight, 0, baseFormat, unpackType, nullptr);
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, i, internalFormat, mipWidth, mipHeight, 0, baseFormat, unpackType, nullptr);
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, i, internalFormat, mipWidth, mipHeight, 0, baseFormat, unpackType, nullptr);
                    mipWidth = std::max<GLsizei>(1, mipWidth / 2);
                    mipHeight = std::max<GLsizei>(1, mipHeight / 2);
                }
            }
            if (target == GL_TEXTURE_2D)
            {
                for (int i = 0; i < levels; ++i)
                {
                    glTexImage2D(GL_TEXTURE_2D, i, internalFormat, mipWidth, mipHeight, 0, baseFormat, unpackType, nullptr);
                    mipWidth = std::max<GLsizei>(1, mipWidth / 2);
                    mipHeight = std::max<GLsizei>(1, mipHeight / 2);
                }
            }
            if (target == GL_TEXTURE_1D_ARRAY)
            {
                for (int i = 0; i < levels; ++i)
                {
                    glTexImage2D(GL_TEXTURE_1D_ARRAY, i, internalFormat, mipWidth, height, 0, baseFormat, unpackType, nullptr);
                    mipWidth = std::max<GLsizei>(1, mipWidth / 2);
                }
            }
        }
        else
        {
            if (target == GL_TEXTURE_1D)
            {
                for (int i = 0; i < levels; ++i)
                {
                    glTexImage1D(GL_TEXTURE_1D, i, internalFormat, mipWidth, 0, baseFormat, unpackType, nullptr);
                    mipWidth = std::max<GLsizei>(1, mipWidth / 2);
                }
            }
        }
    }
    auto image            = new GLImage(context, desc);
    image->m_Impl->resId  = resIdx;
    return image;
}

RTLib::Ext::GL::GLImage::~GLImage() noexcept
{
    m_Impl.reset();
}

void RTLib::Ext::GL::GLImage::Destroy() noexcept
{
    if (!m_Impl) { return; }
    if (m_Impl->resId == 0) { return; }
    if (m_Impl->hasOwnership){
        glDeleteTextures(1, &m_Impl->resId);
    }
    m_Impl->resId = 0;
    m_Impl->hasOwnership = false;
}

auto RTLib::Ext::GL::GLImage::GetViewType() const noexcept -> GLImageViewType
{
    return GLImageViewType(m_Impl->viewType);
}

auto RTLib::Ext::GL::GLImage::GetImageType() const noexcept -> GLImageType
{
    return GLImageType(m_Impl->imageType);
}

auto RTLib::Ext::GL::GLImage::GetFormat() const noexcept -> GLFormat
{
    return GLFormat(m_Impl->format);
}

auto RTLib::Ext::GL::GLImage::GetExtent() const noexcept -> GLExtent3D
{
    return GLExtent3D(m_Impl->extent);
}

auto RTLib::Ext::GL::GLImage::GetMipExtent(uint32_t level) const noexcept -> GLExtent3D
{
    level = std::min<size_t>(level, m_Impl->mipLevels);
    auto viewType = m_Impl->viewType;
    auto extent   = m_Impl->extent;
    switch (viewType)
    {
    case RTLib::Ext::GL::GLImageViewType::e1D:
    case RTLib::Ext::GL::GLImageViewType::e1DArray:
        for (auto l = 0; l < level; ++l) {
            extent.width = std::max<uint32_t>(extent.width / 2, 1);
        }
        return { extent.width,0,0 };
        break;
    case RTLib::Ext::GL::GLImageViewType::e2D:
    case RTLib::Ext::GL::GLImageViewType::e2DArray:
    case RTLib::Ext::GL::GLImageViewType::eCubemap:
    case RTLib::Ext::GL::GLImageViewType::eCubemapArray:
        for (auto l = 0; l < level; ++l) {
            extent.width = std::max<uint32_t>( extent.width / 2, 1);
            extent.height = std::max<uint32_t>(extent.height/ 2, 1);
        }
        return { extent.width,extent.height,0 };
        break;
    case RTLib::Ext::GL::GLImageViewType::e3D:
        for (auto l = 0; l < level; ++l) {
            extent.width  = std::max<uint32_t>(extent.width  / 2, 1);
            extent.height = std::max<uint32_t>(extent.height / 2, 1);
            extent.depth  = std::max<uint32_t>(extent.depth  / 2, 1);
        }
        return { extent.width,extent.height, extent.depth };
        break;
    default: return { 0,0,0 };
        break;
    }
}

auto RTLib::Ext::GL::GLImage::GetMipLevels() const noexcept -> uint32_t
{
    return uint32_t(m_Impl->mipLevels);
}

auto RTLib::Ext::GL::GLImage::GetArrayLayers() const noexcept -> uint32_t
{
    return uint32_t(m_Impl->arrayLayers);
}

auto RTLib::Ext::GL::GLImage::GetFlags() const noexcept -> GLImageCreateFlags
{
    return GLImageCreateFlags(m_Impl->flags);
}

RTLib::Ext::GL::GLImage::GLImage(GLContext* context, const GLImageCreateDesc& desc) noexcept:m_Impl{new Impl{context,desc}}
{
}

auto RTLib::Ext::GL::GLImage::GetResId() const noexcept -> GLuint
{
    assert(m_Impl);
    assert(m_Impl->resId);
    return m_Impl->resId;
}
