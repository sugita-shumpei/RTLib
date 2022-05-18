#include "GLContextImpl.h"
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
	glGetIntegerv(GL_MAJOR_VERSION              , &m_Impl->m_GLMajorVersion);
	glGetIntegerv(GL_MINOR_VERSION              , &m_Impl->m_GLMinorVersion);
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK       , &m_Impl->m_GLProfileMask);
	glGetIntegerv(GL_MAX_IMAGE_UNITS            , &m_Impl->m_GLMaxImageUnits);
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

bool RTLib::Ext::GL::GLContext::SupportVersion(uint32_t majorVersion, uint32_t minorVersion) const noexcept
{
	return m_Impl->SupportVersion(majorVersion, minorVersion);
}

auto RTLib::Ext::GL::GLContext::GetMajorVersion() const noexcept -> uint32_t
{
	return m_Impl->m_GLMajorVersion;
}

auto RTLib::Ext::GL::GLContext::GetMinorVersion() const noexcept -> uint32_t
{
	return m_Impl->m_GLMinorVersion;
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
