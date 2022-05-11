#include "ImplGLVertexArray.h"
#include "ImplGLUtils.h"
#include <iostream>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			namespace Internal
			{
				class ImplGLVertexArrayBase : public ImplGLBindableBase
				{
				public:
					friend class ImplGLBindable;

				public:
					virtual ~ImplGLVertexArrayBase() noexcept {}

				protected:
					virtual bool Create() noexcept override
					{
						GLuint resId;
						glGenVertexArrays(1, &resId);
						if (resId == 0)
						{
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void Destroy() noexcept
					{
						GLuint resId = GetResId();
						glDeleteVertexArrays(1, &resId);
						SetResId(0);
					}
					virtual void Bind(GLenum target)
					{
						GLuint resId = GetResId();
						if (resId > 0)
						{
#ifndef NDEBUG
							std::cout << "BIND " << ToString(target) << ": " << GetName() << std::endl;
#endif
							glBindVertexArray(resId);
						}
					}
					virtual void Unbind(GLenum target)
					{
#ifndef NDEBUG
						std::cout << "UNBIND " << ToString(target) << ": " << GetName() << std::endl;
						glBindVertexArray(0);
#endif
					}
				};
			}
		}
	}
}

auto RTLib::Ext::GL::Internal::ImplGLVertexArray::New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, const ImplGLBindingPoint* bpBuffer) -> ImplGLVertexArray*
{
	if (!table || !bPoint || !bpBuffer)
	{
		return nullptr;
	}
	auto vertexArray = new ImplGLVertexArray(table, bPoint, bpBuffer);
	if (vertexArray)
	{
		vertexArray->InitBase<ImplGLVertexArrayBase>();
		auto res = vertexArray->Create();
		if (!res)
		{
			delete vertexArray;
			return nullptr;
		}
	}
	return vertexArray;
}

RTLib::Ext::GL::Internal::ImplGLVertexArray::~ImplGLVertexArray() noexcept {}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::Bind()
{
	return ImplGLBindable::Bind(GL_VERTEX_ARRAY);
}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::IsBindable() const noexcept {
	if (!IsEnabled()) { return false; }
	return ImplGLBindable::IsBindable(GL_VERTEX_ARRAY);
}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::SetVertexAttribBinding(GLuint attribIndex, GLuint bindIndex)
{
	auto resId = GetResId();
	if (!resId  || IsEnabled()) { return false; }
	m_VertexAttribBindings[attribIndex] = bindIndex;
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::SetVertexAttribFormat(GLuint attribIndex, GLint size, GLenum type, GLboolean normalized, GLuint relativeOffset)
{
	auto resId = GetResId();
	if (!resId || IsEnabled()) { return false; }
	m_VertexAttributes[attribIndex].attribIndex    = attribIndex;
	m_VertexAttributes[attribIndex].size           = size;
	m_VertexAttributes[attribIndex].type           = type;
	m_VertexAttributes[attribIndex].normalized     = normalized;
	m_VertexAttributes[attribIndex].relativeOffset = relativeOffset;
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::SetVertexBuffer(GLuint bindIndex, ImplGLBuffer* vertexBuffer, GLsizei stride, GLintptr offset)
{
	auto resId = GetResId();
	if (!resId || IsEnabled() || !vertexBuffer) { return false; }
	if (!vertexBuffer->IsAllocated()) { return false; }
	m_VertexBindings[bindIndex].bindIndex    = bindIndex;
	m_VertexBindings[bindIndex].vertexBuffer = vertexBuffer;
	m_VertexBindings[bindIndex].stride       = stride;
	m_VertexBindings[bindIndex].offset       = offset;
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::SetIndexBuffer(ImplGLBuffer* indexBuffer)
{
	auto resId = GetResId();
	if (!resId || IsEnabled() || !indexBuffer) { return false; }
	if (!indexBuffer->IsAllocated()) { return false; }
	m_IndexBuffer = indexBuffer;
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::Enable()
{
	auto resId = GetResId();
	if (!resId || IsEnabled())     { return false; }
	if (m_VertexAttributes.empty() || m_VertexBindings.empty() || m_VertexAttribBindings.empty()) { return false; }
	if (!GetBindingPoint()->IsBindable(GL_VERTEX_ARRAY))  { return false; }
	if (!m_BPBuffer->IsBindable(GL_ARRAY_BUFFER))         { return false; }
	if (!m_BPBuffer->IsBindable(GL_ELEMENT_ARRAY_BUFFER)) { return false; }
	{
		//VALIDATE ATTRIB 
		for (auto& [attribIndex,tempInfo] : m_VertexAttributes) {
			if (m_VertexAttribBindings.count(attribIndex) == 0) {
				return false;
			}
		}
		for (auto& [attribIndex, bindingIndex] : m_VertexAttribBindings) {
			if (m_VertexBindings.count(bindingIndex) == 0) {
				return false;
			}
		}
	}
	if (m_IndexBuffer) {
		if (!m_IndexBuffer->IsAllocated() ||
			 m_IndexBuffer->IsBinded() ||
			 m_IndexBuffer->IsMapped()) {
			return false;
		}
	}
	for (auto& [index,binding] : m_VertexBindings) 
	{
		if (!binding.vertexBuffer) {
			return false;
		}
		if (!binding.vertexBuffer->IsAllocated() ||
			 binding.vertexBuffer->IsBinded() ||
			 binding.vertexBuffer->IsMapped()) {
			return false;
		}
	}
	glBindVertexArray(resId);
	if (m_IndexBuffer) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBuffer->GetResId());
	}
	else {
#ifdef NDEBUG
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif
	}
	for (auto& [bindingIndex, bindingInfo] : m_VertexBindings) {
		std::vector<GLuint> tVertexAttribIndices = {};
		for (auto& [attribIndex, tBindingIndex] : m_VertexAttribBindings) {
			if (tBindingIndex == bindingIndex) { tVertexAttribIndices.push_back(attribIndex); }
		}
		if (tVertexAttribIndices.empty()) {
			continue;
		}
		glBindBuffer(GL_ARRAY_BUFFER, bindingInfo.vertexBuffer->GetResId());
		for (const auto& attribIndex : tVertexAttribIndices) {
			if (m_VertexAttributes.count(attribIndex) > 0) {
				const auto& attrib = m_VertexAttributes.at(attribIndex);
				uintptr_t offset = bindingInfo.stride * bindingInfo.offset + attrib.relativeOffset;
				glEnableVertexAttribArray(attribIndex);
				glVertexAttribPointer(attribIndex, attrib.size, attrib.type, attrib.normalized, bindingInfo.stride, reinterpret_cast<const void*>(offset));
			}
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	glBindVertexArray(0);
#ifndef NDEBUG
	if (m_IndexBuffer) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
#endif
	m_IsEnabled = true;
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::DrawArrays(GLenum mode, GLsizei count, GLint first) {
	if (!IsValidMode(mode)) {
		return false;
	}
	bool isBindForDraw = false;
	if (!IsBinded()) {
		if (!IsBindable()) {
			return false;
		}
		isBindForDraw = false;
	}
	else {
		isBindForDraw = true;
	}
	if (!isBindForDraw) {
		glBindVertexArray(GetResId());
	}
	glDrawArrays(mode, first, count);
#ifndef NDEBUG
	if (!isBindForDraw) {
		glBindVertexArray(0);
	}
#endif
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLVertexArray::DrawElements(GLenum mode, GLenum type, GLsizei count, uintptr_t indexOffset)
{
	if (!IsValidMode(mode) || !IsValidType(type)|| !IsEnabled() || count == 0) {
		return false;
	}
	bool isBindForDraw = false;
	if (!IsBinded()) {
		if (!IsBindable()) {
			return false;
		}
		isBindForDraw = false;
	}
	else {
		isBindForDraw = true;
	}
	if (!isBindForDraw) {
		glBindVertexArray(GetResId());
	}
	glDrawElements(mode, count, type, reinterpret_cast<void*>(indexOffset));
#ifndef NDEBUG
	if (!isBindForDraw) {
		glBindVertexArray(0);
	}
#endif
	return true;
}

RTLib::Ext::GL::Internal::ImplGLVertexArray::ImplGLVertexArray(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, const ImplGLBindingPoint* bpBuffer) noexcept : ImplGLBindable(table, bPoint), m_BPBuffer{ bpBuffer }{}
