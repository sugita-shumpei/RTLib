#include <RTLib/Ext/GL/GLVertexArray.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLContext.h>
#include "GLTypeConversions.h"
auto RTLib::Ext::GL::GLVertexArray::New(GLContext* context) -> GLVertexArray*
{
    if (!context)
    {
        return nullptr;
    }
    GLuint resId = 0;
    glGenVertexArrays(1,&resId);
    if (!resId){ return nullptr;}
    return new GLVertexArray(context,resId);;
}

RTLib::Ext::GL::GLVertexArray::~GLVertexArray() noexcept {}
void RTLib::Ext::GL::GLVertexArray::Destroy()noexcept{
    if (m_ResId==0){
        return;
    }
    glDeleteVertexArrays(1,&m_ResId);
    m_ResId = 0;
}
bool RTLib::Ext::GL::GLVertexArray::SetVertexAttribBinding(GLuint attribIndex, GLuint bindIndex)
{
    auto resId = GetResId();
    if (!resId  || IsEnabled()) { return false; }
    m_VertexAttribBindings[attribIndex] = bindIndex;
    return true;
}

bool RTLib::Ext::GL::GLVertexArray::SetVertexAttribFormat(GLuint attribIndex, GLVertexFormat format, GLboolean normalized, GLuint relativeOffset)
{auto resId = GetResId();
    if (!resId || IsEnabled()) { return false; }
    
    m_VertexAttributes[attribIndex].attribIndex    = attribIndex;
    m_VertexAttributes[attribIndex].size           = GetGLVertexFormatNumChannels(format);
    m_VertexAttributes[attribIndex].type           = GetGLVertexFormatGLenum(format);
    m_VertexAttributes[attribIndex].normalized     = normalized;
    m_VertexAttributes[attribIndex].relativeOffset = relativeOffset;
    return true;
}

bool RTLib::Ext::GL::GLVertexArray::SetVertexBuffer(GLuint bindIndex, GLBuffer* vertexBuffer, GLsizei stride, GLintptr offset)
{
    auto resId = GetResId();
    if (!resId || IsEnabled() || !vertexBuffer) { return false; }
    m_VertexBindings[bindIndex].bindIndex    = bindIndex;
    m_VertexBindings[bindIndex].vertexBuffer = vertexBuffer;
    m_VertexBindings[bindIndex].stride       = stride;
    m_VertexBindings[bindIndex].offset       = offset;
    return true;
}

bool RTLib::Ext::GL::GLVertexArray::SetIndexBuffer(GLBuffer* indexBuffer)
{
    auto resId = GetResId();
    if (!resId || IsEnabled() || !indexBuffer) { return false; }
    m_IndexBuffer = indexBuffer;
    return true;
}

bool RTLib::Ext::GL::GLVertexArray::Enable()
{
    auto resId = GetResId();
    if (!resId || IsEnabled())     { return false; }
    if (m_VertexAttributes.empty() || m_VertexBindings.empty() || m_VertexAttribBindings.empty()) { return false; }
    {
        //VALIDATE ATTRIB
        for (auto& [attribIndex,tempInfo] : m_VertexAttributes) {
            if (m_VertexAttribBindings.count(attribIndex) == 0) {
                return false;
            }
        }
        for (auto& [attribIndex, bindingIndex] : m_VertexAttribBindings) {
            if (m_VertexBindings.count(bindingIndex) == 0) {
                return false;
            }
        }
    }
    for (auto& [index,binding] : m_VertexBindings)
    {
        if (!binding.vertexBuffer) {
            return false;
        }
    }
    {
        m_Context->SetVertexArray(this);
        if (m_IndexBuffer) {
            m_Context->SetBuffer(GLBufferUsageIndex, m_IndexBuffer);
        }
        else {
            m_Context->InvalidateBuffer(GLBufferUsageIndex);
        }
        for (auto& [bindingIndex, bindingInfo] : m_VertexBindings) {
            std::vector<GLuint> tVertexAttribIndices = {};
            for (auto& [attribIndex, tBindingIndex] : m_VertexAttribBindings) {
                if (tBindingIndex == bindingIndex) { tVertexAttribIndices.push_back(attribIndex); }
            }
            if (tVertexAttribIndices.empty()) {
                continue;
            }
            m_Context->SetBuffer(GLBufferUsageVertex, bindingInfo.vertexBuffer);
            for (const auto& attribIndex : tVertexAttribIndices) {
                if (m_VertexAttributes.count(attribIndex) > 0) {
                    const auto& attrib = m_VertexAttributes.at(attribIndex);
                    uintptr_t offset = bindingInfo.stride * bindingInfo.offset + attrib.relativeOffset;
                    glEnableVertexAttribArray(attribIndex);
                    glVertexAttribPointer(attribIndex, attrib.size, attrib.type, attrib.normalized, bindingInfo.stride, reinterpret_cast<const void*>(offset));
                }
            }
        }
        m_Context->InvalidateVertexArray();
        if (m_IndexBuffer) {
            m_Context->InvalidateBuffer(GLBufferUsageIndex);
        }
    }
    m_IsEnabled = true;
    return true;
}

RTLib::Ext::GL::GLVertexArray::GLVertexArray(GLContext* context, GLuint resId)noexcept:m_Context(context),m_ResId(resId){}

bool RTLib::Ext::GL::GLVertexArray::IsEnabled() const noexcept
{
    return m_IsEnabled;
}