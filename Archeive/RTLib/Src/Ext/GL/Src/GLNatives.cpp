#include <RTLib/Ext/GL/GLNatives.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLProgram.h>
#include <RTLib/Ext/GL/GLProgramPipeline.h>
#include <RTLib/Ext/GL/GLShader.h>
#include <RTLib/Ext/GL/GLVertexArray.h>
#include <RTLib/Ext/GL/GLFramebuffer.h>



auto RTLib::Ext::GL::GLNatives::GetResId(GLBuffer* buf) -> GLuint
{
	return buf ? buf->GetResId() : 0;
}

auto RTLib::Ext::GL::GLNatives::GetResId(GLImage* img) -> GLuint
{
	return img ? img->GetResId() : 0;
}

auto RTLib::Ext::GL::GLNatives::GetResId(GLTexture* tex) -> GLuint
{
	return tex ? tex->GetResId() : 0;
}

auto RTLib::Ext::GL::GLNatives::GetResId(GLProgram* prg) -> GLuint
{
	return prg ? prg->GetResId() : 0;
}

auto RTLib::Ext::GL::GLNatives::GetResId(GLProgramPipeline* prg) -> GLuint
{
	return prg ? prg->GetResId() : 0;
}

auto RTLib::Ext::GL::GLNatives::GetResId(GLShader* shd) -> GLuint
{
	return shd ? shd->GetResId() : 0;
}

auto RTLib::Ext::GL::GLNatives::GetResId(GLVertexArray* vao) -> GLuint
{
	return vao ? vao->GetResId() : 0;
}

auto RTLib::Ext::GL::GLNatives::GetResId(GLFramebuffer* fbo) -> GLuint
{
	return GLuint();
}

auto RTLib::Ext::GL::GLNatives::GetResId(GLRenderbuffer* rbo) -> GLuint
{
	return GLuint();
}
