#include "GLContextImpl.h"
#include <cassert>
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
	if (!srcBuffer || !dstBuffer) { return false; }
	if (SupportVersion(4, 5)) {
		for (const auto& region : regions) {
			glCopyNamedBufferSubData(srcBuffer->GetResId(), dstBuffer->GetResId(), region.srcOffset, region.dstOffset, region.size);
		}
	}
	else {
		auto srcTarget = srcBuffer->GetCurrentTarget();
		auto dstTarget = dstBuffer->GetCurrentTarget();
		if (!srcTarget) {
			srcTarget  = GL_COPY_READ_BUFFER;
			glBindBuffer(GL_COPY_READ_BUFFER, srcBuffer->GetResId());
		}
		if (!dstTarget) {
			dstTarget = GL_COPY_WRITE_BUFFER;
			glBindBuffer(GL_COPY_WRITE_BUFFER, dstBuffer->GetResId());
		}
		for (const auto& region : regions) {
			glCopyBufferSubData(*srcTarget,*dstTarget, region.srcOffset, region.dstOffset, region.size);
		}
	}
	return true;
}

bool RTLib::Ext::GL::GLContext::CopyMemoryToBuffer(GLBuffer* buffer, const std::vector<GLMemoryBufferCopy>& regions)
{
	if (!buffer) { return false; }
	if (!(buffer->GetMemoryProperty() & GLMemoryPropertyHostWrite)) {
		return false;
	}
	char* mappedData = nullptr;
	if (SupportVersion(4, 5)) {
		for (const auto& region : regions) {
			mappedData = (char*)glMapNamedBufferRange(buffer->GetResId(), region.dstOffset, region.size, GL_MAP_WRITE_BIT);
			std::memcpy(mappedData, region.srcData, region.size);
			assert(glUnmapNamedBuffer(buffer->GetResId()));
		}
	}
	else {
		auto curTarget = buffer->GetCurrentTarget();
		if (!curTarget) {
			glBindBuffer(GL_COPY_READ_BUFFER, buffer->GetResId());
			curTarget =  GL_COPY_READ_BUFFER;
		}
		for (const auto& region : regions) {
			mappedData = (char*)glMapBufferRange(*curTarget, region.dstOffset, region.size, GL_MAP_WRITE_BIT);
			std::memcpy(mappedData, region.srcData, region.size);
			assert(glUnmapBuffer(*curTarget));
		}
	}
	return true;
}

bool RTLib::Ext::GL::GLContext::CopyBufferToMemory(GLBuffer* buffer, const std::vector<GLBufferMemoryCopy>& regions)
{
	if (!buffer) { return false; }
	if (!(buffer->GetMemoryProperty() & GLMemoryPropertyHostRead)) {
		return false;
	}
	char* mappedData = nullptr;
	if (SupportVersion(4, 5)) {
		for (const auto& region : regions) {
			mappedData = (char*)glMapNamedBufferRange(buffer->GetResId(), region.srcOffset, region.size, GL_MAP_READ_BIT);
			std::memcpy(region.dstData, mappedData, region.size);
			assert(glUnmapNamedBuffer(buffer->GetResId()));
		}
	}
	else {
		auto curTarget = buffer->GetCurrentTarget();
		if (!curTarget) {
			glBindBuffer(GL_COPY_READ_BUFFER, buffer->GetResId());
			curTarget = GL_COPY_READ_BUFFER;
		}
		for (const auto& region : regions) {
			mappedData = (char*)glMapBufferRange(*curTarget, region.srcOffset, region.size, GL_MAP_READ_BIT);
			std::memcpy(region.dstData, mappedData, region.size);
			assert(glUnmapBuffer(*curTarget));
		}
	}
	return true;
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

auto RTLib::Ext::GL::GLContext::CreateBuffer(const GLBufferCreateDesc& desc) -> GLBuffer*
{
	return GLBuffer::Allocate(this,desc);
}

auto RTLib::Ext::GL::GLContext::CreateTexture(const GLTextureCreateDesc& desc) -> GLTexture*
{
	return nullptr;
}
