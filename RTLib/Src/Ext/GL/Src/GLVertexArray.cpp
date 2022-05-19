#include <RTLib/Ext/GL/GLVertexArray.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLContext.h>
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

bool RTLib::Ext::GL::GLVertexArray::SetVertexAttribFormat(GLuint attribIndex, GLint size, GLenum type, GLboolean normalized, GLuint relativeOffset)
{
    auto resId = GetResId();
    if (!resId || IsEnabled()) { return false; }
    m_VertexAttributes[attribIndex].attribIndex    = attribIndex;
    m_VertexAttributes[attribIndex].size           = size;
    m_VertexAttributes[attribIndex].type           = type;
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
    glBindVertexArray(resId);
    if (m_IndexBuffer) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBuffer->GetResId());
    }
    else {
#ifdef NDEBUG
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif
    }
    for (auto& [bindingIndex, bindingInfo] : m_VertexBindings) {
        std::vector<GLuint> tVertexAttribIndices = {};
        for (auto& [attribIndex, tBindingIndex] : m_VertexAttribBindings) {
            if (tBindingIndex == bindingIndex) { tVertexAttribIndices.push_back(attribIndex); }
        }
        if (tVertexAttribIndices.empty()) {
            continue;
        }
        glBindBuffer(GL_ARRAY_BUFFER, bindingInfo.vertexBuffer->GetResId());
        for (const auto& attribIndex : tVertexAttribIndices) {
            if (m_VertexAttributes.count(attribIndex) > 0) {
                const auto& attrib = m_VertexAttributes.at(attribIndex);
                uintptr_t offset = bindingInfo.stride * bindingInfo.offset + attrib.relativeOffset;
                glEnableVertexAttribArray(attribIndex);
                glVertexAttribPointer(attribIndex, attrib.size, attrib.type, attrib.normalized, bindingInfo.stride, reinterpret_cast<const void*>(offset));
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    glBindVertexArray(0);
#ifndef NDEBUG
    if (m_IndexBuffer) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
#endif
    m_IsEnabled = true;
    return true;
}

bool RTLib::Ext::GL::GLVertexArray::DrawArrays(GLenum mode, GLsizei count, GLint first) {
    if (!IsValidMode(mode)) {
        return false;
    }
    bool isBindForDraw = false;
    glBindVertexArray(GetResId());
    glDrawArrays(mode, first, count);
    return true;
}

bool RTLib::Ext::GL::GLVertexArray::DrawElements(GLenum mode, GLenum type, GLsizei count, uintptr_t indexOffset)
{
    if (!IsValidMode(mode) || !IsValidType(type)|| !IsEnabled() || count == 0) {
        return false;
    }
    glBindVertexArray(GetResId());
    glDrawElements(mode, count, type, reinterpret_cast<void*>(indexOffset));
    return true;
}

RTLib::Ext::GL::GLVertexArray::GLVertexArray(GLContext* context, GLuint resId)noexcept:m_Context(context),m_ResId(resId){}
