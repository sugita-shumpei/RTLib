#include "ImplGLBuffer.h"
#include "ImplGLUtils.h"
#include <iostream>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLBufferBase : public ImplGLBindableBase {
				public:
					friend class ImplGLBindable;
				public:
					virtual ~ImplGLBufferBase()noexcept {}
					bool IsMapped()const noexcept { return m_IsMapped; }
					bool MapMemory(GLenum target, void** pMappedData, GLenum access) {
						if (IsMapped()) {
							return false;
						}
						*pMappedData = glMapBuffer(target, access);
						m_IsMapped = true;
						return true;
					}
					bool MapMemory(GLenum target, void** pMappedData, GLenum access, GLsizeiptr offset, GLsizeiptr size) {
						if (IsMapped()) {
							return false;
						}
						*pMappedData = glMapBufferRange(target, offset, size, access);
						m_IsMapped = true;
						return true;
					}
					bool UnmapMemory(GLenum target) {
						if (!IsMapped()) {
							return false;
						}
						glUnmapBuffer(target);
						m_IsMapped = false;
						return true;
					}
				protected:
					virtual bool      Create()noexcept override {
						GLuint resId;
						glGenBuffers(1, &resId);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						m_IsMapped = false;
						return true;
					}
					virtual void     Destroy()noexcept {
						GLuint resId = GetResId();
						glDeleteBuffers(1, &resId);
						SetResId(0);
					}
					virtual void   Bind(GLenum target) {
						GLuint resId = GetResId();
						if (resId > 0) {
#ifndef NDEBUG
							std::cout << "BIND " << ToString(target) << ": " << GetName() << std::endl;
#endif
							glBindBuffer(target, resId);
						}
					}
					virtual void Unbind(GLenum target) {
						UnmapMemory(target);
						glBindBuffer(target, 0);
					}
				private:
					bool m_IsMapped = false;
				};
			}
		}
	}
}
auto RTLib::Ext::GL::Internal::ImplGLBuffer::New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, GLenum defaultTarget) -> ImplGLBuffer* {
	if (!table || !bPoint) {
		return nullptr;
	}
	if (!bPoint->HasTarget(defaultTarget)) { return false; }
	auto buffer = new ImplGLBuffer(table, bPoint, defaultTarget);
	if (buffer) {
		buffer->InitBase<ImplGLBufferBase>();
		auto res = buffer->Create();
		if (!res) {
			delete buffer;
			return nullptr;
		}
	}
	return buffer;
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::Bind()
{
	return ImplGLBindable::Bind(m_DefaultTarget);
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::Bind(GLenum target) {
	return ImplGLBindable::Bind(target);
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::Allocate(GLenum usage, size_t size, const void* pInitialData)
{
	if (IsAllocated()) {
		return false;
	}
	bool bindForAllocated = false;
	if (IsBinded()) {
		if (GetBindedTarget() != GetDefTarget()) {
			return false;
		}
	}
	else {
		bindForAllocated = true;
	}
	if (bindForAllocated) {
		if (!Bind()) { return false; }
	}
	glBufferData(*GetBindedTarget(), size, pInitialData, usage);
	m_AllocationInfo = AllocationInfo{ size, usage };
	if (bindForAllocated) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyFromMemory(const void* pSrcData, size_t size, size_t offset)
{
	if (!pSrcData || !this->IsAllocated() || IsMapped()) {
		return false;
	}
	bool isBindedCopyDst  = false;
	auto targetForCopyDst = GetBindedTarget();
	if (!targetForCopyDst) {
		 targetForCopyDst = GetBindableTargetForCopyDst();
		 if (!targetForCopyDst) { return false; }
	}
	else {
		isBindedCopyDst = true;
	}
	if (!isBindedCopyDst) {
		glBindBuffer(*targetForCopyDst, GetResId());
	}
	glBufferSubData( *targetForCopyDst, offset, size, pSrcData);
	if (!isBindedCopyDst) {
		glBindBuffer(*targetForCopyDst, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyToMemory(void* pDstData, size_t size, size_t offset)
{
	if (!pDstData || !this->IsAllocated() || IsMapped()) {
		return false;
	}
	bool isBindedCopySrc = false;
	auto targetForCopySrc= GetBindedTarget();
	if (!targetForCopySrc) {
		targetForCopySrc = GetBindableTargetForCopySrc();
		if (!targetForCopySrc) { return false; }
	}
	else {
		isBindedCopySrc  = true;
	}
	if ( !isBindedCopySrc) {
		glBindBuffer(*targetForCopySrc, GetResId());
	}
	glGetBufferSubData(*targetForCopySrc, offset, size, pDstData);
	if ( !isBindedCopySrc) {
		glBindBuffer(*targetForCopySrc, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyFromBuffer(ImplGLBuffer* srcBuffer, size_t size, size_t srcOffset, size_t dstOffset)
{
	return CopyBuffer2Buffer(srcBuffer, this, size, srcOffset, dstOffset);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyToBuffer(  ImplGLBuffer* dstBuffer, size_t size, size_t dstOffset, size_t srcOffset)
{
	return CopyBuffer2Buffer(this, dstBuffer, size, srcOffset, dstOffset);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::MapMemory(void** pMappedData, GLenum access)
{
	auto base = GetBase();
	if (!IsAllocated()||!pMappedData || !base) {
		return false;
	}
	GLenum target;
	if (!IsBinded()) {
		if (Bind()) {
			target = GetDefTarget();
		}
		else {
			if (access == GL_READ_ONLY) {
				if (!Bind(GL_COPY_READ_BUFFER)) {
					return false;
				}
				target = GL_COPY_READ_BUFFER;
			}
			else if (access == GL_WRITE_ONLY) {
				if (!Bind(GL_COPY_WRITE_BUFFER)) {
					return false;
				}
				target = GL_COPY_WRITE_BUFFER;
			}
			else {
				return false;
			}
		}
	}
	else {
		target = *GetBindedTarget();
	}
	return static_cast<ImplGLBufferBase*>(base)->MapMemory(target,pMappedData, access);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::MapMemory(void** pMappedData, GLenum access, GLsizeiptr offset, GLsizeiptr size)
{
	auto base = GetBase();
	if (!IsAllocated() || !pMappedData || !base) {
		return false;
	}
	GLenum target;
	if (!IsBinded()) {
		if (Bind()) {
			target = GetDefTarget();
		}
		else {
			if (access & GL_MAP_READ_BIT == GL_MAP_READ_BIT) {
				if (!Bind(GL_COPY_READ_BUFFER)) {
					return false;
				}
				target = GL_COPY_READ_BUFFER;
			}
			else if (access & GL_MAP_WRITE_BIT == GL_MAP_WRITE_BIT) {
				if (!Bind(GL_COPY_WRITE_BUFFER)) {
					return false;
				}
				target = GL_COPY_WRITE_BUFFER;
			}
			else {
				return false;
			}
		}
	}
	else {
		target = *GetBindedTarget();
	}
	return static_cast<ImplGLBufferBase*>(base)->MapMemory(target, pMappedData, access, offset, size);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::UnmapMemory()
{
	auto base = GetBase();
	if (!IsAllocated() || !IsBinded() || !base) {
		return false;
	}
	return static_cast<ImplGLBufferBase*>(base)->UnmapMemory(*GetBindedTarget());
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::IsMapped() const noexcept { auto base = GetBase(); return base ? static_cast<const ImplGLBufferBase*>(base)->IsMapped() : false; }

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyBuffer2Buffer(ImplGLBuffer* srcBuffer, ImplGLBuffer* dstBuffer, size_t size, size_t srcOffset, size_t dstOffset)
{
	if (!srcBuffer || !dstBuffer) {
		return false;
	}
	if (!srcBuffer->IsAllocated() || srcBuffer->IsMapped()) {
		return false;
	}
	if (!dstBuffer->IsAllocated() || dstBuffer->IsMapped()) {
		return false;
	}

	bool isBindedCopySrc = false;
	auto targetForCopySrc= srcBuffer->GetBindedTarget();

	bool isBindedCopyDst = false;
	auto targetForCopyDst= dstBuffer->GetBindedTarget();
	bool isSameDefTarget = srcBuffer->GetDefTarget() == dstBuffer->GetDefTarget();
	if (!targetForCopySrc) {
		targetForCopySrc = srcBuffer->GetBindableTargetForCopySrc();
		if (!targetForCopySrc) {
			return false;
		}
	}
	else {
		isBindedCopySrc = true;
	}
	if (!targetForCopyDst) {
		if (isSameDefTarget) {
			if (dstBuffer->IsBindable(GL_COPY_WRITE_BUFFER)) {
				targetForCopyDst = GL_COPY_WRITE_BUFFER;
			}
			else {
				return false;
			}
		}
		else {
			targetForCopyDst = srcBuffer->GetBindableTargetForCopyDst();
			if (!targetForCopyDst) {
				return false;
			}
		}
	}
	else {
		isBindedCopyDst = true;
	}
	if (!isBindedCopySrc) {
		glBindBuffer(*targetForCopySrc, srcBuffer->GetResId());
	}
	if (!isBindedCopyDst) {
		glBindBuffer(*targetForCopyDst, dstBuffer->GetResId());
	}
	glCopyBufferSubData(*targetForCopySrc, *targetForCopyDst, srcOffset, dstOffset, size);
	if (!isBindedCopySrc) {
		glBindBuffer(*targetForCopySrc, 0);
	}
	if (!isBindedCopyDst) {
		glBindBuffer(*targetForCopyDst, 0);
	}
	return true;
}

auto RTLib::Ext::GL::Internal::ImplGLBuffer::GetBindableTargetForCopySrc() const noexcept -> std::optional<GLenum>
{
	if (IsBindable(m_DefaultTarget)) {
		return m_DefaultTarget;
	}
	if (IsBindable(GL_COPY_READ_BUFFER)) {
		return GL_COPY_READ_BUFFER;
	}
	return std::optional<GLenum>();
}

auto RTLib::Ext::GL::Internal::ImplGLBuffer::GetBindableTargetForCopyDst() const noexcept -> std::optional<GLenum>
{
	if (IsBindable(m_DefaultTarget)) {
		return m_DefaultTarget;
	}
	if (IsBindable(GL_COPY_WRITE_BUFFER)) {
		return GL_COPY_WRITE_BUFFER;
	}
	return std::optional<GLenum>();
}
