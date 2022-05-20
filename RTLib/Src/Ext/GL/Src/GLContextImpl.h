#ifndef RTLIB_EXT_GL_GL_CONTEXT_IMPL_H
#define RTLIB_EXT_GL_GL_CONTEXT_IMPL_H
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLVertexArray.h>
#include <RTLib/Ext/GL/GLShader.h>
#include <RTLib/Ext/GL/GLProgram.h>
struct RTLib::Ext::GL::GLContext::Impl
{
	GLint m_GLMajorVersion  = 0;
	GLint m_GLMinorVersion  = 0;
	GLint m_GLProfileMask   = 0;
	GLint m_GLMaxImageUnits = 0;
	bool  m_IsInitialized   = false;
    GLVertexArray* m_VAO = nullptr;
    GLProgram* m_Program = nullptr;
	bool SupportVersion(uint32_t versionMajor, uint32_t versionMinor)const noexcept
	{
		if (m_GLMajorVersion > versionMajor) {
			return true;
		}
		else if (m_GLMajorVersion < versionMajor) {
			return false;
		}
		return m_GLMinorVersion >= versionMinor;
	}
};
#endif
