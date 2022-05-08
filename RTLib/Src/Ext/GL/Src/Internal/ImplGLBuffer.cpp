#include "ImplGLBuffer.h"

bool RTLib::Ext::GL::Internal::ImplGLBuffer::Allocate(GLenum target, GLenum usage, size_t size, const void* pInitialData)
{
	if (IsAllocated()) {
		return false;
	}
	bool bindForAllocated = false;
	if (IsBinded()) {
		if (GetTarget() != target) {
			return false;
		}
	}
	else {
		bindForAllocated = true;
	}
	if (bindForAllocated) {
		if (!Bind(target)) { return false; }
	}
	glBufferData(*GetTarget(), size, pInitialData, usage);
	m_AllocationInfo = AllocationInfo{ size, usage };
	if (bindForAllocated) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyFromMemory(const void* pSrcData, size_t size, size_t offset)
{
	if (!pSrcData || IsMapped()) {
		return false;
	}
	bool isBinded = IsBinded();
	bool bindedForCopy = false;
	GLenum target;
	if (!isBinded) {
		if (GetBindingPoint()->HasBindable(GL_COPY_WRITE_BUFFER)) {
			return nullptr;
		}
		bindedForCopy = true;
		target = GL_COPY_WRITE_BUFFER;
	}
	else {
		target = *GetTarget();
	}
	if (bindedForCopy) {
		glBindBuffer(target, GetResId());
	}
	glBufferSubData(target, offset, size, pSrcData);
	if (bindedForCopy) {
		glBindBuffer(target, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyToMemory(void* pDstData, size_t size, size_t offset)
{
	if (!pDstData || !this->IsAllocated() || IsMapped()) {
		return false;
	}
	bool isBinded = IsBinded();
	bool bindedForCopy = false;
	GLenum target;
	if (!isBinded) {
		if (GetBindingPoint()->HasBindable(GL_COPY_READ_BUFFER)) {
			return nullptr;
		}
		bindedForCopy = true;
		target = GL_COPY_READ_BUFFER;
	}
	else {
		target = *GetTarget();
	}
	if (bindedForCopy) {
		glBindBuffer(target, GetResId());
	}
	glGetBufferSubData(target, offset, size, pDstData);
	if (bindedForCopy) {
		glBindBuffer(target, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyFromBuffer(ImplGLBuffer* srcBuffer, size_t size, size_t srcOffset, size_t dstOffset)
{
	if (!srcBuffer||!this->IsAllocated() || IsMapped()) {
		return false;
	}
	if (!srcBuffer->IsAllocated()) {
		return false;
	}
	auto* dstBuffer = this;
	bool bindedForCopySrc = false;
	bool bindedForCopyDst = false;
	GLenum srcTarget;
	GLenum dstTarget;
	if (!srcBuffer->IsBinded()) {
		if (srcBuffer->GetBindingPoint()->HasBindable(GL_COPY_READ_BUFFER)) {
			return false;
		}
		bindedForCopySrc = true;
		srcTarget = GL_COPY_READ_BUFFER;
	}
	else {
		srcTarget = *srcBuffer->GetTarget();
	}
	if (!dstBuffer->IsBinded()) {
		if (dstBuffer->GetBindingPoint()->HasBindable(GL_COPY_WRITE_BUFFER)) {
			return false;
		}
		bindedForCopyDst = true;
		dstTarget = GL_COPY_WRITE_BUFFER;
	}
	else {
		dstTarget = *dstBuffer->GetTarget();
	}
	if (bindedForCopySrc) {
		glBindBuffer(srcTarget,srcBuffer->GetResId());
	}
	if (bindedForCopyDst) {
		glBindBuffer(dstTarget,dstBuffer->GetResId());
	}
	glCopyBufferSubData(srcTarget, dstTarget, srcOffset, dstOffset, size);
	if (bindedForCopySrc) {
		glBindBuffer(srcTarget, 0);
	}
	if (bindedForCopyDst) {
		glBindBuffer(dstTarget, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyToBuffer(ImplGLBuffer* dstBuffer, size_t size, size_t dstOffset, size_t srcOffset)
{
	if (!dstBuffer || !this->IsAllocated() || IsMapped()) {
		return false;
	}
	if (!dstBuffer->IsAllocated()) {
		return false;
	}
	auto* srcBuffer = this;
	bool bindedForCopySrc = false;
	bool bindedForCopyDst = false;
	GLenum srcTarget;
	GLenum dstTarget;
	if (!srcBuffer->IsBinded()) {
		if (srcBuffer->GetBindingPoint()->HasBindable(GL_COPY_READ_BUFFER)) {
			return false;
		}
		bindedForCopySrc = true;
		srcTarget = GL_COPY_READ_BUFFER;
	}
	else {
		srcTarget = *srcBuffer->GetTarget();
	}
	if (!dstBuffer->IsBinded()) {
		if (dstBuffer->GetBindingPoint()->HasBindable(GL_COPY_WRITE_BUFFER)) {
			return false;
		}
		bindedForCopyDst = true;
		dstTarget = GL_COPY_WRITE_BUFFER;
	}
	else {
		dstTarget = *dstBuffer->GetTarget();
	}
	if (bindedForCopySrc) {
		glBindBuffer(srcTarget, srcBuffer->GetResId());
	}
	if (bindedForCopyDst) {
		glBindBuffer(dstTarget, dstBuffer->GetResId());
	}
	glCopyBufferSubData(srcTarget, dstTarget, srcOffset, dstOffset, size);
	if (bindedForCopySrc) {
		glBindBuffer(srcTarget, 0);
	}
	if (bindedForCopyDst) {
		glBindBuffer(dstTarget, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::MapMemory(void** pMappedData, GLenum access)
{
	auto base = GetBase();
	if (!IsAllocated()||!pMappedData || !base) {
		return false;
	}
	GLenum target;
	if (!IsBinded()) {
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
	else {
		target = *GetTarget();
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
	else {
		target = *GetTarget();
	}
	return static_cast<ImplGLBufferBase*>(base)->MapMemory(target, pMappedData, access, offset, size);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::UnmapMemory()
{
	bool isBinded = IsBinded();
	auto base = GetBase();
	if (!IsAllocated() || !isBinded||!base) {
		return false;
	}
	return static_cast<ImplGLBufferBase*>(base)->UnmapMemory(*GetTarget());
}
