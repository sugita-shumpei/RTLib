#include "GLContextState.h"

void RTLib::Ext::GL::GLContextState::SetProgram(const GLBaseProgram& obj)
{
	glUseProgram(static_cast<GLuint>(obj));
}

void RTLib::Ext::GL::GLContextState::SetProgramPipeline(const GLBaseProgramPipeline& obj)
{
	glBindProgramPipeline(static_cast<GLuint>(obj));
}

void RTLib::Ext::GL::GLContextState::BindBuffer(GLenum target, const GLBaseBuffer& obj, bool forVaoReset)
{
	glBindBuffer(target,static_cast<GLuint>(obj));
}

void RTLib::Ext::GL::GLContextState::BindUniformBuffer(int32_t index, const GLBaseBuffer& obj, GLintptr offset, GLsizeiptr size)
{
	glBindBufferRange(GL_UNIFORM_BUFFER, index, static_cast<GLuint>(obj), offset,size);
}

void RTLib::Ext::GL::GLContextState::BindStorageBuffer(int32_t index, const GLBaseBuffer& obj, GLintptr offset, GLsizeiptr size)
{
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, index, static_cast<GLuint>(obj), offset, size);
}

void RTLib::Ext::GL::GLContextState::SetActiveTexture(int32_t index)
{
	glActiveTexture(GL_TEXTURE0+index);
}

void RTLib::Ext::GL::GLContextState::BindTexture(int32_t index, GLenum target, const GLBaseTexture& tex)
{
	glBindTexture(target,static_cast<GLuint>(tex));
}

void RTLib::Ext::GL::GLContextState::BindSampler(uint32_t index, const GLBaseSampler& smp)
{
	glBindSampler(index, static_cast<GLuint>(smp));
}

void RTLib::Ext::GL::GLContextState::BindVertexArray(const GLBaseVertexArray& vao)
{
	glBindVertexArray(static_cast<GLuint>(vao));
}

void RTLib::Ext::GL::GLContextState::BindFramebuffer(const GLBaseFramebuffer& fbo)
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(fbo));
	glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(fbo));
}

void RTLib::Ext::GL::GLContextState::BindRenderbuffer(const GLBaseRenderbuffer& rbo)
{
	glBindRenderbuffer(GL_RENDERBUFFER, static_cast<GLuint>(rbo));
}
