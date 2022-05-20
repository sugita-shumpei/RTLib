#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <optional>
#include "GLContextImpl.h"
#include "GLObjectBase.h"
#include "GLTypeConversions.h"
struct RTLib::Ext::GL::GLBuffer::Impl
{
	Impl(GLContext* ctx, const GLBufferCreateDesc& desc)noexcept :base(true), usage{ desc.usage }, size{desc.size},access{ desc.access }, context{ ctx } {}
	GLContext*			  context;
    GLBaseBuffer          base   ;
	size_t				  size   ;
	GLBufferUsageFlags    usage  ;
	GLMemoryPropertyFlags access ;
};

auto RTLib::Ext::GL::GLBuffer::Allocate(GLContext* context, const GLBufferCreateDesc& desc)->GLBuffer* {
    auto res        = new GLBuffer(context,desc);
	auto mainTarget = RTLib::Ext::GL::GetGLBufferMainUsageTarget(desc.usage);
	auto accessMode = RTLib::Ext::GL::GetGLBufferUsageFlagsAccessMode(desc.access);
	auto accessFreq = RTLib::Ext::GL::GetGLBufferCreateDescBufferAccessFrequency(desc);
	auto usageCount = RTLib::Ext::GL::GetGLBufferUsageCount(desc.usage);
	glBindBuffer(mainTarget, res->GetResId());
	if ((usageCount == 1)&& context->SupportVersion(4, 4)) {
		glBufferStorage(mainTarget, desc.size, desc.pData, accessMode);
	}
	else {
		glBufferData(mainTarget, desc.size, desc.pData, accessFreq);
	}
    return res;
}
    RTLib::Ext::GL::GLBuffer::~GLBuffer()noexcept {
}
void RTLib::Ext::GL::GLBuffer::Destroy() {
	if (!m_Impl) {
		return;
	}
    m_Impl->base.Destroy();
	m_Impl.reset();
}
auto RTLib::Ext::GL::GLBuffer::GetBufferUsage()const noexcept -> GLBufferUsageFlags {
	return m_Impl ? m_Impl->usage:0;

}
RTLib::Ext::GL::GLBuffer::GLBuffer(GLContext* context, const GLBufferCreateDesc& desc)noexcept :m_Impl{ new Impl(context,desc) } {
	
}
auto RTLib::Ext::GL::GLBuffer::GetResId()const noexcept -> GLuint {
	return m_Impl ? m_Impl->base.GetObjectID() : 0;
}

auto RTLib::Ext::GL::GLBuffer::GetCurrentTarget() const noexcept -> std::optional<GLenum>
{
	return std::optional<GLenum>();
}

auto RTLib::Ext::GL::GLBuffer::GetMemoryProperty() const noexcept -> GLMemoryPropertyFlags
{
	return m_Impl ? m_Impl->access : 0;
}
