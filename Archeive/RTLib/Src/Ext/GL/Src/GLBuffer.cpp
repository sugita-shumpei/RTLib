#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <optional>
#include "GLContextImpl.h"
#include "GLObjectBase.h"
#include "GLTypeConversions.h"
struct RTLib::Ext::GL::GLBuffer::Impl
{
	Impl(GLContext* ctx, const GLBufferCreateDesc& desc)noexcept :base(true), usage{ desc.usage }, size{desc.size},access{ desc.access }, context{ ctx } {
		mainUsage = RTLib::Ext::GL::GLBufferUsageFlagsUtils::GetMainUsage(desc.usage);
	}
	GLContext*			  context;
    GLBaseBuffer          base   ;
	size_t				  size   ;
	GLBufferUsageFlagBits mainUsage;
	GLBufferUsageFlags    usage  ;
	GLMemoryPropertyFlags access ;
};

auto RTLib::Ext::GL::GLBuffer::Allocate(GLContext* context, const GLBufferCreateDesc& desc)->GLBuffer* {
    auto buffer     = new GLBuffer(context,desc);
	auto usageCount = RTLib::Ext::GL::GLBufferUsageFlagsUtils::GetUsageCount(desc.usage);
	auto mainTarget = RTLib::Ext::GL::GetGLBufferMainUsageTarget(buffer->m_Impl->mainUsage);
	auto accessMode = RTLib::Ext::GL::GetGLBufferUsageFlagsAccessMode(desc.access);
	auto accessFreq = RTLib::Ext::GL::GetGLBufferCreateDescBufferAccessFrequency(desc);
	context->SetBuffer(buffer->m_Impl->mainUsage, buffer);
	if ((usageCount == 1)&& context->SupportVersion(4, 4)) {
		glBufferStorage(mainTarget, desc.size, desc.pData, accessMode);
	}
	else {
		glBufferData(mainTarget, desc.size, desc.pData, accessFreq);
	}
    return buffer;
}
    RTLib::Ext::GL::GLBuffer::~GLBuffer()noexcept {
}
void RTLib::Ext::GL::GLBuffer::Destroy() noexcept{
	if (!m_Impl) {
		return;
	}
    m_Impl->base.Destroy();
}
auto RTLib::Ext::GL::GLBuffer::GetUsages()const noexcept -> GLBufferUsageFlags {
	return m_Impl ? m_Impl->usage:0;

}
auto RTLib::Ext::GL::GLBuffer::GetSizeInBytes() const noexcept -> size_t
{
	return m_Impl ? m_Impl->size : 0;
}
RTLib::Ext::GL::GLBuffer::GLBuffer(GLContext* context, const GLBufferCreateDesc& desc)noexcept :m_Impl{ new Impl(context,desc) } {
	
}
auto RTLib::Ext::GL::GLBuffer::GetMainUsage() const noexcept -> GLBufferUsageFlagBits
{
	return m_Impl ? m_Impl->mainUsage : GLBufferUsageUnknown;
}
auto RTLib::Ext::GL::GLBuffer::GetResId()const noexcept -> GLuint {
	return m_Impl ? m_Impl->base.GetObjectID() : 0;
}

auto RTLib::Ext::GL::GLBuffer::GetMemoryProperty() const noexcept -> GLMemoryPropertyFlags
{
	return m_Impl ? m_Impl->access : 0;
}
RTLib::Ext::GL::GLBufferView::GLBufferView() noexcept:m_Base{nullptr},m_SizeInBytes{0},m_OffsetInBytes{0}{}
RTLib::Ext::GL::GLBufferView::GLBufferView(GLBuffer* base, size_t offsetInBytes, size_t sizeInBytes)noexcept {
	m_Base = base;
	m_SizeInBytes = std::min<size_t>(sizeInBytes, base->GetSizeInBytes());
	m_OffsetInBytes = offsetInBytes;
}
RTLib::Ext::GL::GLBufferView::GLBufferView(GLBuffer* base)noexcept :GLBufferView( base,0, base->GetSizeInBytes() ) {
}
RTLib::Ext::GL::GLBufferView::GLBufferView(const GLBufferView& bufferView, size_t offsetInBytes, size_t sizeInBytes)noexcept 
: GLBufferView(bufferView.m_Base,bufferView.m_OffsetInBytes+offsetInBytes,bufferView.m_SizeInBytes+sizeInBytes) {

}
RTLib::Ext::GL::GLBufferView::GLBufferView(const GLBufferView& bufferView)noexcept
 :GLBufferView(bufferView.m_Base, bufferView.m_OffsetInBytes, bufferView.m_SizeInBytes) {

}
auto RTLib::Ext::GL::GLBufferView::operator=(const GLBufferView& bufferView)noexcept->GLBufferView& {
	if (this != &bufferView) {
		m_Base = bufferView.m_Base;
		m_OffsetInBytes = bufferView.m_OffsetInBytes;
		m_SizeInBytes = bufferView.m_SizeInBytes;
	}
	return *this;
}