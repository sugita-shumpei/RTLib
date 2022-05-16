#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLContext.h>
#include "GLTypeConversions.h"

auto RTLib::Ext::GL::GLBuffer::Allocate(GLContext* context, const GLBufferDesc& desc) -> GLBuffer*
{
    GLuint resId = 0;
    glGenBuffers(1, &resId);
    if (!resId) { return nullptr; }
    auto usageCount = GetGLBufferUsageCount(desc.usage);
    auto mainUsage  = GetGLBufferMainUsage( desc.usage);
    auto mainTarge  = GetGLBufferMainUsageTarget(mainUsage);
    
    return nullptr;
}

void RTLib::Ext::GL::GLBuffer::Destroy()
{
    if (m_ResId) {
        glDeleteBuffers(1, &m_ResId);
    }
    m_ResId = 0;
}

RTLib::Ext::GL::GLBuffer::GLBuffer(GLuint resId, GLBufferUsageFlags usage) noexcept :m_ResId{ resId }, m_Usage{ usage }
{
}
