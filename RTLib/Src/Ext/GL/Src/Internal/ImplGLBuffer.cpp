#include "ImplGLBuffer.h"
#include "ImplGLBuffer.h"
#include "ImplGLBuffer.h"
#include "ImplGLBuffer.h"
#include "ImplGLBuffer.h"
#include "ImplGLBuffer.h"

bool RTLib::Ext::GL::Internal::ImplGLBuffer::Allocate(GLenum usage, size_t size, const void* pInitialData)
{
	if (!IsBinded() || IsAllocated()) {
		return false;
	}
	glBufferData(*GetTarget(), size, pInitialData, usage);
	m_AllocationInfo = AllocationInfo{ size, usage };
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyImageFromMemory(const void* pSrcData, size_t size, size_t offset)
{
	if (!pSrcData) {
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

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyImageToMemory(void* pDstData, size_t size, size_t offset)
{
	if (!pDstData) {
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

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyImageFromBuffer(ImplGLBuffer* srcBuffer, size_t size, size_t srcOffset, size_t dstOffset)
{
	if (!srcBuffer||!this->IsAllocated()) {
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

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyImageToBuffer(ImplGLBuffer* dstBuffer, size_t size, size_t dstOffset, size_t srcOffset)
{
	if (!dstBuffer || !this->IsAllocated()) {
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
	bool isBinded = IsBinded();
	if (!isBinded||!pMappedData || IsMapped()) {
		return false;
	}
	*pMappedData = glMapBuffer(*GetTarget(), access);
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::MapMemory(void** pMappedData, GLenum access, GLsizeiptr offset, GLsizeiptr size)
{
	bool isBinded = IsBinded();
	if (!isBinded || !pMappedData || IsMapped()) {
		return false;
	}
	*pMappedData = glMapBufferRange(*GetTarget(),offset,size,access);
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::UnmapMemory()
{
	bool isBinded = IsBinded();
	if (!isBinded ||!IsMapped()) {
		return false;
	}
	return glUnmapBuffer(*GetTarget());
}
