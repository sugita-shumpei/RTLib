#ifndef RTLIB_EXT_GL_GL_CONTEXT_IMPL_H
#define RTLIB_EXT_GL_GL_CONTEXT_IMPL_H
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLVertexArray.h>
#include <RTLib/Ext/GL/GLShader.h>
#include <RTLib/Ext/GL/GLProgram.h>
#include <RTLib/Ext/GL/GLProgramPipeline.h>
#include <RTLib/Core/Utility.h>
#include <array>
struct RTLib::Ext::GL::GLContext::Impl
{
	enum GLBufferUsageIndex:uint32_t
	{
		GLBufferUsageIndexVertex            = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageVertex)),
		GLBufferUsageIndexAtomicCounter     = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageAtomicCounter)),
		GLBufferUsageIndexDispatchIndirect  = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageDispatchIndirect)),
		GLBufferUsageIndexDrawIndirect      = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageDrawIndirect)),
		GLBufferUsageIndexIndex             = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageIndex)) ,
		GLBufferUsageIndexQuery             = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageQuery)) ,
		GLBufferUsageIndexStorage           = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageStorage)) ,
		GLBufferUsageIndexTexture           = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageTexture)) ,
		GLBufferUsageIndexTransformFeedback = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageTransformFeedback)),
		GLBufferUsageIndexUniform           = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageUniform)) ,
		GLBufferUsageIndexImageCopySrc      = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageImageCopySrc)),
		GLBufferUsageIndexImageCopyDst      = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageImageCopyDst)),
		GLBufferUsageIndexGenericCopySrc    = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageGenericCopySrc)),
		GLBufferUsageIndexGenericCopyDst    = RTLib::Core::Utility::Log2(static_cast<uint64_t>(RTLib::Ext::GL::GLBufferUsageGenericCopyDst)),
		GLBufferUsageIndexCount,
	};
	using GLBuffers  = std::array<  GLBuffer*, GLBufferUsageIndexCount>;
	using GLImages   = std::vector< GLImage*>;
	struct BufferRange {
		GLBuffer* buffer;
		size_t    offset;
		size_t    size;
	};

	GLint          m_GLMajorVersion  = 0;
	GLint          m_GLMinorVersion  = 0;
	GLint          m_GLProfileMask   = 0;
	GLint          m_GLMaxImageUnits = 0;
	bool           m_IsInitialized   = false;
	//STATE
	GLint          m_ActiveTexUnit   = 0;
	GLBuffers      m_Buffers         = {};
	GLImages       m_Images          = {};
    GLVertexArray* m_VertexArray     = nullptr;
	GLProgram    * m_Program         = nullptr;
	GLProgramPipeline* m_ProgramPipeline = nullptr;

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
