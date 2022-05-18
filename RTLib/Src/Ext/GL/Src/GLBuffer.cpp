#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLContext.h>
#include "GLContextImpl.h"
#include "GLTypeConversions.h"
struct RTLib::Ext::GL::GLBuffer::Impl {
    Impl(GLContext* ctx, GLuint res, GLBufferUsageFlags usg, GLMemoryPropertyFlags props, size_t siz)noexcept :context{ ctx }, usage{ usg }, size{ siz },memoryProps{props}, resId{res}{}
    GLContext*                 context;
    GLuint                       resId;
    GLBufferUsageFlags           usage;
    GLMemoryPropertyFlags  memoryProps;
    size_t                        size;
};

auto RTLib::Ext::GL::GLBuffer::Allocate(GLContext* context, const GLBufferCreateDesc& desc) -> GLBuffer*
{
    GLuint resId = 0;
    glGenBuffers(1, &resId);
    if (!resId) { return nullptr; }
    auto usageCount = GetGLBufferUsageCount(desc.usage);
    auto mainUsage  = GetGLBufferMainUsage( desc.usage);
    auto mainTarget = GetGLBufferMainUsageTarget(mainUsage);
    auto buffer     = new GLBuffer(context,resId, desc);
    auto state      = context->GetContextState();
    if ((usageCount == 1)) {
        if (context->SupportVersion(4, 5)) {
            glNamedBufferStorage(resId, desc.size, desc.pData, GL_DYNAMIC_STORAGE_BIT);
        }
        else if (context->SupportVersion(4, 4)) {
            glBindBuffer(mainTarget, resId);
            glBufferStorage(mainTarget, desc.size, desc.pData, GL_DYNAMIC_STORAGE_BIT);
        }
        else {
            glBindBuffer(mainTarget, resId);
            glBufferData(mainTarget, desc.size, desc.pData, GL_STATIC_DRAW);
        }
    }
    else {
        glBindBuffer(mainTarget, resId);
        glBufferData(mainTarget, desc.size, desc.pData, GL_STATIC_DRAW);
    }
    return nullptr;
}

void RTLib::Ext::GL::GLBuffer::Destroy()
{
    if (!m_Impl) { return; }
    m_Impl.reset();
}

RTLib::Ext::GL::GLBuffer::GLBuffer(GLContext* context, GLuint resId, const GLBufferCreateDesc& desc)noexcept
    :m_Impl{ new Impl(context,resId,desc.usage,desc.props,desc.size) } {}

auto RTLib::Ext::GL::GLBuffer::GetResId() const noexcept -> GLuint
{
    return m_Impl?m_Impl->resId:0;
}

auto RTLib::Ext::GL::GLBuffer::GetBufferUsage() const noexcept -> GLBufferUsageFlags
{
    return m_Impl?m_Impl->usage:0;
}
