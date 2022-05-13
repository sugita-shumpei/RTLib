#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <GLBufferBindingPoint.h>

struct RTLib::Ext::GL::GLContext::Impl
{
	GLint m_GLVersionMajor = 0;
	GLint m_GLVersionMinor = 0;
	GLint m_GLProfileMask = 0;
	GLint m_GLMaxImageUnits = 0;
	bool  m_IsInitialized = false;
	std::unique_ptr<Internals::GLBufferBindingPoint> m_BufferBP = nullptr;

	bool IsSupportedVersion(uint32_t versionMajor, uint32_t versionMinor)const noexcept
	{
		if (m_GLVersionMajor > versionMajor) {
			return true;
		}
		else if (m_GLVersionMajor < versionMajor) {
			return false;
		}
		return m_GLVersionMinor >= versionMinor;
	}
};
RTLib::Ext::GL::GLContext:: GLContext() noexcept {
	m_Impl = std::unique_ptr<Impl>(new Impl());
}

RTLib::Ext::GL::GLContext::~GLContext() noexcept {
	m_Impl.reset();
}

bool RTLib::Ext::GL::GLContext::Initialize()
{
	if (!m_Impl) {
		return false;
	}
	if (m_Impl->m_IsInitialized) {
		return false;
	}
	if (!InitLoader()) {
		return false;
	}
	glGetIntegerv(GL_MAJOR_VERSION              , &m_Impl->m_GLVersionMajor);
	glGetIntegerv(GL_MINOR_VERSION              , &m_Impl->m_GLVersionMinor);
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK       , &m_Impl->m_GLProfileMask);
	glGetIntegerv(GL_MAX_IMAGE_UNITS            , &m_Impl->m_GLMaxImageUnits);

	m_Impl->m_BufferBP = std::unique_ptr<RTLib::Ext::GL::Internals::GLBufferBindingPoint>(new RTLib::Ext::GL::Internals::GLBufferBindingPoint());
	
	if (m_Impl->IsSupportedVersion(2, 0)) {

		m_Impl->m_BufferBP->AddValidTarget(GL_ARRAY_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_ELEMENT_ARRAY_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_COPY_WRITE_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_DRAW_INDIRECT_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_PIXEL_PACK_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_PIXEL_UNPACK_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_TRANSFORM_FEEDBACK_BUFFER);
	}
	if (m_Impl->IsSupportedVersion(3, 0)) {
		GLint maxTransformFeedbackBufferBindings = 0;
		if (m_Impl->IsSupportedVersion(4, 0))
		{
			glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_BUFFERS, &maxTransformFeedbackBufferBindings);
		}
		else {
			glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS, &maxTransformFeedbackBufferBindings);
		}
		m_Impl->m_BufferBP->AddValidTargetRange(GL_TRANSFORM_FEEDBACK_BUFFER, maxTransformFeedbackBufferBindings);
	}
	if (m_Impl->IsSupportedVersion(3, 1)) {
		GLint maxUniformBufferBindings = 0;
		glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &maxUniformBufferBindings);

		m_Impl->m_BufferBP->AddValidTarget(GL_COPY_READ_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_UNIFORM_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_TEXTURE_BUFFER);

		m_Impl->m_BufferBP->AddValidTargetRange(GL_UNIFORM_BUFFER, maxUniformBufferBindings);
	}
	if (m_Impl->IsSupportedVersion(4, 2)){
		GLint maxAtomicCounterBufferBindings = 0;
		glGetIntegerv(GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS, &maxAtomicCounterBufferBindings);

		m_Impl->m_BufferBP->AddValidTarget(GL_ATOMIC_COUNTER_BUFFER);

		m_Impl->m_BufferBP->AddValidTargetRange(GL_ATOMIC_COUNTER_BUFFER, maxAtomicCounterBufferBindings);
	}
	if (m_Impl->IsSupportedVersion(4, 3)) {
		GLint maxShaderStorageBufferBindings = 0;
		glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxShaderStorageBufferBindings);

		m_Impl->m_BufferBP->AddValidTarget(GL_DISPATCH_INDIRECT_BUFFER);
		m_Impl->m_BufferBP->AddValidTarget(GL_SHADER_STORAGE_BUFFER);

		m_Impl->m_BufferBP->AddValidTargetRange(GL_SHADER_STORAGE_BUFFER, maxShaderStorageBufferBindings);
	}
	if (m_Impl->IsSupportedVersion(4, 4)) {
		m_Impl->m_BufferBP->AddValidTarget(GL_QUERY_BUFFER);
	}

	m_Impl->m_IsInitialized = true;
	return true;
}

void RTLib::Ext::GL::GLContext::Terminate()
{
	if (!m_Impl->m_IsInitialized) {
		return;
	}
	FreeLoader();
	m_Impl->m_IsInitialized = false;
}

bool RTLib::Ext::GL::GLContext::CopyBuffer(GLBuffer* srcBuffer, GLBuffer* dstBuffer, const std::vector<GLBufferCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyMemoryToBuffer(GLBuffer* buffer, const std::vector<GLMemoryBufferCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyBufferToMemory(GLBuffer* buffer, const std::vector<GLBufferMemoryCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyImageToBuffer(GLImage* srcImage, GLBuffer* dstBuffer, const std::vector<GLBufferImageCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyBufferToImage(GLBuffer* srcBuffer, GLImage* dstImage, const std::vector<GLBufferImageCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyImageToMemory(GLImage* image, const std::vector<GLImageMemoryCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyMemoryToImage(GLImage* image, const std::vector<GLImageMemoryCopy>& regions)
{
	return false;
}
