#include "ImplGLContext.h"
#include "ImplGLBuffer.h"
#include "ImplGLTexture.h"
#include "ImplGLFramebuffer.h"
#include "ImplGLRenderbuffer.h"
#include "ImplGLSampler.h"
auto RTLib::Ext::GL::Internal::ImplGLContext::CreateBuffer(GLenum defaultTarget) -> ImplGLBuffer*
{
    return ImplGLBuffer::New(&m_ResourceTable,&m_BPBuffer, &m_BPBufferRange, defaultTarget);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateTexture(GLenum target) -> ImplGLTexture*
{
    return ImplGLTexture::New(target,&m_ResourceTable, &m_BPTexture, &m_BPBuffer);
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
    return ImplGLVertexArray::New(&m_ResourceTable, &m_BPVertexArray, &m_BPBuffer);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateShader(GLenum shaderT) -> ImplGLShader*
{
    switch (shaderT) {

    case GL_VERTEX_SHADER:
        return ImplGLVertexShader::New(&m_ResourceTable, IsSpirvSupported());
    case GL_FRAGMENT_SHADER:
        return ImplGLFragmentShader::New(&m_ResourceTable, IsSpirvSupported());
    case GL_GEOMETRY_SHADER:
        return ImplGLGeometryShader::New(&m_ResourceTable, IsSpirvSupported());
    case GL_TESS_EVALUATION_SHADER:
        return ImplGLTesselationEvaluationShader::New(&m_ResourceTable, IsSpirvSupported());
    case GL_TESS_CONTROL_SHADER:
        return ImplGLTesselationControlShader::New(&m_ResourceTable, IsSpirvSupported());
    case GL_COMPUTE_SHADER:
        if (!IsSupportedVersion(4, 3)) {
            return nullptr;
        }
        return ImplGLComputeShader::New(&m_ResourceTable, IsSpirvSupported());
    default:
        return nullptr;
    }
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateGraphicsProgram() -> ImplGLGraphicsProgram*
{
    return ImplGLGraphicsProgram::New(&m_ResourceTable, &m_ProgramSlot);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateComputeProgram() -> ImplGLComputeProgram*
{
    if (!IsSupportedVersion(4, 3)) { return nullptr; }
    return ImplGLComputeProgram::New(&m_ResourceTable, &m_ProgramSlot);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateSeparateProgram() -> ImplGLSeparateProgram*
{
    if (!IsSupportedVersion(4, 1)) { return nullptr; }
    return ImplGLSeparateProgram::New(&m_ResourceTable,&m_ProgramSlot);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateGraphicsProgramPipeline() -> ImplGLGraphicsProgramPipeline*
{
    if (!IsSupportedVersion(4, 1)) { return nullptr; }
    return ImplGLGraphicsProgramPipeline::New(&m_ResourceTable, &m_BPProgramPipeline, &m_ProgramSlot);
}

auto RTLib::Ext::GL::Internal::ImplGLContext::CreateComputeProgramPipeline() -> ImplGLComputeProgramPipeline*
{
    if (!IsSupportedVersion(4, 1)) { return nullptr; }
    return ImplGLComputeProgramPipeline::New(&m_ResourceTable, &m_BPProgramPipeline, &m_ProgramSlot);
}
