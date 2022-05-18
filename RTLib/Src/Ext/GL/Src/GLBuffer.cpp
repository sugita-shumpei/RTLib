#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <optional>
#include "GLContextImpl.h"
#include "GLTypeConversions.h"
struct RTLib::Ext::GL::GLBuffer::Impl
{
	Impl(GLContext* ctx, GLuint rid, const GLBufferCreateDesc& desc)noexcept :resId{ rid }, usage{ desc.usage }, size{desc.size},access{ desc.access }, context{ ctx } {}
	GLContext*			  context;
	GLuint                resId  ;
	size_t				  size   ;
	GLBufferUsageFlags    usage  ;
	GLMemoryPropertyFlags access ;
};

auto RTLib::Ext::GL::GLBuffer::Allocate(GLContext* context, const GLBufferCreateDesc& desc)->GLBuffer* {
	GLuint resId = 0;
	glGenBuffers(1, &resId);
	auto mainTarget = RTLib::Ext::GL::GetGLBufferMainUsageTarget(desc.usage);
	auto accessMode = RTLib::Ext::GL::GetGLBufferUsageFlagsAccessMode(desc.access);
	auto accessFreq = RTLib::Ext::GL::GetGLBufferCreateDescBufferAccessFrequency(desc);
	auto usageCount = RTLib::Ext::GL::GetGLBufferUsageCount(desc.usage);
	glBindBuffer(mainTarget, resId);
	if ((usageCount == 1)&& context->SupportVersion(4, 4)) {
		glBufferStorage(mainTarget, desc.size, desc.pData, accessMode);
	}
	else {
		glBufferData(mainTarget, desc.size, desc.pData, accessFreq);
	}
	return new GLBuffer(context,resId,desc);
}
    RTLib::Ext::GL::GLBuffer::~GLBuffer()noexcept {

}
void RTLib::Ext::GL::GLBuffer::Destroy() {
	if (!m_Impl) {
		return;
	}
	glDeleteBuffers(1, &m_Impl->resId);
	m_Impl.reset();
}
auto RTLib::Ext::GL::GLBuffer::GetBufferUsage()const noexcept -> GLBufferUsageFlags {
	return m_Impl ? m_Impl->usage:0;

}
RTLib::Ext::GL::GLBuffer::GLBuffer(GLContext* context, GLuint resId, const GLBufferCreateDesc& desc)noexcept :m_Impl{ new Impl(context,resId,desc) } {
	
}
auto RTLib::Ext::GL::GLBuffer::GetResId()const noexcept -> GLuint {
	return m_Impl ? m_Impl->resId : 0;
}

auto RTLib::Ext::GL::GLBuffer::GetCurrentTarget() const noexcept -> std::optional<GLenum>
{
	return std::optional<GLenum>();
}

auto RTLib::Ext::GL::GLBuffer::GetMemoryProperty() const noexcept -> GLMemoryPropertyFlags
{
	return m_Impl ? m_Impl->access : 0;
}