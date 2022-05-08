#include "ImplGLContext.h"
#include "ImplGLBuffer.h"
#include "ImplGLTexture.h"
#include "ImplGLFramebuffer.h"
#include "ImplGLRenderbuffer.h"
#include "ImplGLSampler.h"
auto RTLib::Ext::GL::Internal::ImplGLContext::CreateBuffer() -> ImplGLBuffer*
{
    return ImplGLBuffer::New(&m_ResourceTable,&m_BPBuffer);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateTexture(GLenum target) -> ImplGLTexture*
{
    return ImplGLTexture::New(target,&m_ResourceTable, &m_BPTexture,&m_BPBuffer);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateSampler() -> ImplGLSampler*
{
    return ImplGLSampler::New(&m_ResourceTable, &m_BPSampler);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateFramebuffer() -> ImplGLFramebuffer*
{
    return ImplGLFramebuffer::New(&m_ResourceTable, &m_BPFramebuffer);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateRenderbuffer() -> ImplGLRenderbuffer*
{
    return ImplGLRenderbuffer::New(&m_ResourceTable, &m_BPRenderbuffer);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateVertexArray() -> ImplGLVertexArray*
{
    return ImplGLVertexArray::New(&m_ResourceTable, &m_BPRenderbuffer);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateShader(GLenum shaderT) -> ImplGLShader*
{
    return ImplGLShader::New(&m_ResourceTable, shaderT);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateProgram() -> ImplGLProgram*
{
    return ImplGLProgram::New(&m_ResourceTable);
}

