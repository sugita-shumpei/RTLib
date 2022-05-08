#include "ImplGLTexture.h"
#include "ImplGLBuffer.h"
#include "ImplGLUtils.h"
bool RTLib::Ext::GL::Internal::ImplGLTexture::Allocate(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth)
{
	if (IsAllocated()|| width == 0|| height == 0|| depth == 0 || !m_BPBuffer) {
		return false;
	}
	if (m_BPBuffer->HasBindable(GL_PIXEL_UNPACK_BUFFER)) {
		return false;
	}
	switch (m_Target) {
	case GL_TEXTURE_1D:
		if (layers !=1||height != 1 || depth != 1) {
			return false;
		}
		return AllocateTexture1D(internalFormat, levels, width);
	case GL_TEXTURE_2D:
		if (layers != 1 || depth != 1) {
			return false;
		}
		return AllocateTexture2D(internalFormat, levels, width, height);
	case GL_TEXTURE_3D:
		if (layers != 1) {
			return false;
		}
		return AllocateTexture3D(internalFormat, levels, width, height, depth);
	case GL_TEXTURE_1D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		return AllocateTexture1DArray(internalFormat, levels, layers, width);
	case GL_TEXTURE_2D_ARRAY:
		if (depth != 1) {
			return false;
		}
		return AllocateTexture2DArray(internalFormat, levels, layers, width, height);
	case GL_TEXTURE_RECTANGLE:
		if (layers != 1 || depth != 1 || levels != 1) {
			return false;
		}
		return AllocateTexture2D(internalFormat, 1, width, height);
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layers != 1 || depth != 1) {
			return false;
		}
		return AllocateTexture2D(internalFormat, levels, width, height);
	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		if (depth != 1) {
			return false;
		}
		return AllocateTexture2DArray(internalFormat, levels, layers, width, height);
	case GL_TEXTURE_CUBE_MAP:
		if (layers != 1 || depth != 1) {
			return false;
		}
		return AllocateTextureCubemap(internalFormat, levels, width, height);
	case GL_TEXTURE_CUBE_MAP_ARRAY:
		if (depth != 1) {
			return false;
		}
		return AllocateTextureCubemapArray(internalFormat, levels,layers, width, height);
	default:
		return false;
	}
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImageFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset)
{
	if (!pData || !IsAllocated() || width == 0 || height == 0 || depth == 0 || level < 0 || !m_BPBuffer) {
		return false;
	}
	if (m_BPBuffer->HasBindable(GL_PIXEL_UNPACK_BUFFER)) {
		return false;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}
	switch (m_Target) {
	case GL_TEXTURE_1D:
		if (layer != 0 || layers!=1 || height!=1 || depth != 1) {
			return false;
		}
		return CopyImage1DFromMemory(pData, format, type, level, width, dstXOffset);
	case GL_TEXTURE_2D:
		if (layer != 0 || layers != 1 || depth != 1) {

			return false;
		}
		return CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
	case GL_TEXTURE_3D:
		if (layer != 0 || layers != 1) {
			return false;
		}
		return CopyImage3DFromMemory(pData, format, type, level, width, height, depth, dstXOffset, dstYOffset, dstZOffset);
	case GL_TEXTURE_1D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		return CopyLayeredImage1DFromMemory(pData, format, type, level, layer, layers, width, dstXOffset);
	case GL_TEXTURE_2D_ARRAY:
		if (depth != 1) {
			return false;
		}
		return CopyLayeredImage2DFromMemory(pData, format, type, level, layer, layers, width, height, dstXOffset, dstYOffset);
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		return CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		return CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		return CopyLayeredImage2DFromMemory(pData, format, type, level, layer, layers, width, height, dstXOffset, dstYOffset);
	default:
		return false;
	}
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImageToMemory(void* pData, GLenum format, GLenum type, GLint level)
{
	if (!pData || !IsAllocated() || level < 0) {
		return false;
	}
	if (m_BPBuffer->HasBindable(GL_PIXEL_PACK_BUFFER)) {
		return false;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}
	GLsizei pixelSize = GetGLFormatTypeSize(m_AllocationInfo->internalFormat, type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		if (!Bind()) {
			return false;
		}
	}
	glPixelStorei(GL_PACK_ALIGNMENT, alignment);
	glGetTexImage(m_Target, level, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImageFromBuffer(ImplGLBuffer* src, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset, GLintptr srcOffset)
{
	if (!IsAllocated() || width == 0 || height == 0 || depth == 0 || level < 0 || !m_BPBuffer || !src) {
		return false;
	}
	if (!src->IsAllocated()) {
		return false;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}
	bool bindForCopy = false;
	if (src->IsBinded()) {
		if (src->GetTarget() != GL_PIXEL_UNPACK_BUFFER) {
			return false;
		}
	}
	else {
		if (m_BPBuffer->HasBindable(GL_PIXEL_UNPACK_BUFFER)) {
			return false;
		}
		bindForCopy = true;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}
	void* pData = reinterpret_cast<void*>(srcOffset);
	bool  res = false;
	switch (m_Target) {
	case GL_TEXTURE_1D:
		if (layer != 0 || layers != 1 || height != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyImage1DFromMemory(pData, format, type, level, width, dstXOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_2D:
		if (layer != 0 || layers != 1 || depth != 1) {

			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_3D:
		if (layer != 0 || layers != 1) {
			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyImage3DFromMemory(pData, format, type, level, width, height, depth, dstXOffset, dstYOffset, dstZOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_1D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyLayeredImage1DFromMemory(pData, format, type, level, layer, layers, width, dstXOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_2D_ARRAY:
		if (depth != 1) {

			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyLayeredImage2DFromMemory(pData, format, type, level, layer, layers, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyLayeredImage2DFromMemory(pData, format, type, level, layer, layers, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	default:
		return false;
	}
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImageToBuffer(ImplGLBuffer* dst, GLenum format, GLenum type, GLint level, GLintptr dstOffset)
{
	if (!dst || !IsAllocated() || level < 0) {

		return false;
	}
	if (!dst->IsAllocated()) {
		return false;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}
	GLsizei pixelSize = GetGLFormatTypeSize(m_AllocationInfo->internalFormat, type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	auto src = this;
	bool bindedForCopySrc = !src->IsBinded();
	bool bindedForCopyDst = false;
	if (dst->IsBinded()){
		if (dst->GetTarget() != GL_PIXEL_PACK_BUFFER) {
			return false;
		}
	}
	else {
		if (dst->GetBindingPoint()->HasBindable(GL_PIXEL_PACK_BUFFER)) {
			return false;
		}
		bindedForCopyDst = true;
	}
	bool sucessForSrcBinded = true;
	bool sucessForDstBinded = true;
	bool result = false;
	if (bindedForCopySrc) {
		sucessForSrcBinded = src->Bind();
	}
	if (bindedForCopyDst) {
		sucessForDstBinded = dst->Bind(GL_PIXEL_PACK_BUFFER);
	}
	if (sucessForSrcBinded && sucessForDstBinded) {
		glPixelStorei(GL_PACK_ALIGNMENT, alignment);
		glGetTexImage(m_Target, level, format, type, reinterpret_cast<void*>(dstOffset));
		result = true;
	}
	else {
		result = false;
	}
	if (bindedForCopySrc) {
		src->Unbind();
	}
	if (bindedForCopyDst) {
		dst->Unbind();
	}
	return result;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImageFromMemory(GLenum target, const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset)
{
	if (!pData || !IsAllocated() || width == 0 || height == 0 || depth == 0 || level < 0 || !m_BPBuffer || !IsCubeFaceTarget(target)) {
		return false;
	}
	if (m_BPBuffer->HasBindable(GL_PIXEL_UNPACK_BUFFER)) {
		return false;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}
	auto txTarget = GetTxTarget();
	if (txTarget == GL_TEXTURE_CUBE_MAP) {
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		return CopyFaceImage2DFromMemory(target, pData, format, type, level, width, height, dstXOffset, dstYOffset);
	}
	if (txTarget == GL_TEXTURE_CUBE_MAP_ARRAY) {
		if (depth != 1) {
			return false;
		}
		return CopyLayeredFaceImage2DFromMemory(target, pData, format, type, level, layer, layers, width, height, dstXOffset, dstYOffset);
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImageToMemory(GLenum target, void* pData, GLenum format, GLenum type, GLint level)
{
	if (!pData || !IsAllocated() || level < 0 || !IsCubeFaceTarget(target) || (GetTxTarget() != GL_TEXTURE_CUBE_MAP && GetTxTarget() != GL_TEXTURE_CUBE_MAP_ARRAY)) {
		return false;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}
	if (m_BPBuffer->HasBindable(GL_PIXEL_PACK_BUFFER)) {
		return false;
	}
	GLsizei pixelSize = GetGLFormatTypeSize(m_AllocationInfo->internalFormat, type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		if (!Bind()) {
			return false;
		}
	}
	glPixelStorei(GL_PACK_ALIGNMENT, alignment);
	glGetTexImage(target, level, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImageFromBuffer(GLenum target, ImplGLBuffer* src, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset, GLintptr srcOffset)
{
	if (!IsAllocated() || width == 0 || height == 0 || depth == 0 || level < 0 || !m_BPBuffer || !src || !IsCubeFaceTarget(target)) {
		return false;
	}
	if (!src->IsAllocated()) {
		return false;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}
	bool bindForCopy = false;
	if (src->IsBinded()) {
		if (src->GetTarget() != GL_PIXEL_UNPACK_BUFFER) {
			return false;
		}
	}
	else {
		if (m_BPBuffer->HasBindable(GL_PIXEL_UNPACK_BUFFER)) {
			return false;
		}
		bindForCopy = true;
	}
	void* pData = reinterpret_cast<void*>(srcOffset);
	bool  res = false;
	auto txTarget = GetTxTarget();
	if (txTarget == GL_TEXTURE_CUBE_MAP) {
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyFaceImage2DFromMemory(target, pData, format, type, level, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) {
			src->Unbind();
		}
		return res;
	}
	if (txTarget == GL_TEXTURE_CUBE_MAP_ARRAY) {
		if (depth != 1) {
			return false;
		}
		if (bindForCopy) {
			if (!src->Bind(GL_PIXEL_UNPACK_BUFFER)) { return false; }
		}
		res = CopyLayeredFaceImage2DFromMemory(target, pData, format, type, level, layer, layers, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) {
			src->Unbind();
		}
		return res;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImageToBuffer(GLenum target, ImplGLBuffer* dst, GLenum format, GLenum type, GLint level, GLintptr srcOffset)
{
	if (!dst || !IsAllocated() || level < 0 || !IsCubeFaceTarget(target) || (GetTxTarget() != GL_TEXTURE_CUBE_MAP && GetTxTarget() != GL_TEXTURE_CUBE_MAP_ARRAY)) {
		return false;
	}
	if (!dst->IsAllocated()) {
		return false;
	}
	if (level >= m_AllocationInfo->levels) {
		return false;
	}

	bool bindedForSrcCopy = !IsBinded();
	bool bindedForDstCopy = false;
	if (dst->IsBinded()) {
		if (dst->GetTarget() != GL_PIXEL_PACK_BUFFER) {
			return false;
		}
	}
	else {
		if (m_BPBuffer->HasBindable(GL_PIXEL_PACK_BUFFER)) {
			return false;
		}
		bindedForDstCopy = true;
	}
	GLsizei pixelSize = GetGLFormatTypeSize(m_AllocationInfo->internalFormat, type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	void* pData = reinterpret_cast<void*>(srcOffset);
	bool successForSrcBind = true;
	bool successForDstBind = true;
	bool result = false;
	if (bindedForSrcCopy) {
		successForSrcBind = Bind();
	}
	if (bindedForDstCopy) {
		successForDstBind = dst->Bind(GL_PIXEL_UNPACK_BUFFER);
	}
	if (successForSrcBind && successForDstBind) {
		glPixelStorei(GL_PACK_ALIGNMENT, alignment);
		glGetTexImage(target, level, format, type, pData);
		result = true;
	}
	if (bindedForSrcCopy) {
		Unbind();
	}
	if (bindedForDstCopy) {
		dst->Unbind();
	}
	return result;
}

auto RTLib::Ext::GL::Internal::ImplGLTexture::GetBPBuffer() const noexcept -> const ImplGLBindingPoint*
{
	return m_BPBuffer;
}

auto RTLib::Ext::GL::Internal::ImplGLTexture::GetMipWidth(GLint level) const noexcept -> GLsizei
{
	if (!IsAllocated()) { return 1; }
	auto width = m_AllocationInfo->width;
	for (int l = 0; l < level; ++l) {
		width = std::max(width / 2, 1);
	}
	return width;
}

auto RTLib::Ext::GL::Internal::ImplGLTexture::GetMipHeight(GLint level) const noexcept -> GLsizei
{
	if (!IsAllocated()) { return 1; }
	auto height = m_AllocationInfo->height;
	for (int l = 0; l < level; ++l) {
		height = std::max(height / 2, 1);
	}
	return height;
}

auto RTLib::Ext::GL::Internal::ImplGLTexture::GetMipDepth(GLint level) const noexcept -> GLsizei
{
	if (!IsAllocated()) { return 1; }
	auto depth = m_AllocationInfo->depth;
	for (int l = 0; l < level; ++l) {
		depth = std::max(depth / 2, 1);
	}
	return depth;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture1D(GLenum internalFormat, GLint levels, GLsizei width)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { if (!Bind()) { return false; } }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage1D(m_Target, level, internalFormat, tempWidth, 0, baseFormat, GL_UNSIGNED_BYTE, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,1,1 };
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture2D(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	auto tempHeight = height;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { if (!Bind()) { return false; } }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage2D(m_Target, level, internalFormat, tempWidth, tempHeight, 0, baseFormat, type, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
		tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,1 };
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture3D(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height, GLsizei depth)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	auto tempHeight = height;
	auto tempDepth = depth;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { if (!Bind()) { return false; } }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage3D(m_Target, level, internalFormat, tempWidth, tempHeight, tempDepth, 0, baseFormat, type, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
		tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		tempDepth = std::max<GLsizei>(tempDepth / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,depth };
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture1DArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { if (!Bind()) { return false; } }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage2D(m_Target, level, internalFormat, tempWidth, layers, 0, baseFormat, type, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,layers,width,1,1 };
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture2DArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	auto tempHeight = height;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { if (!Bind()) { return false; } }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage3D(m_Target, level, internalFormat, tempWidth, tempHeight, layers, 0, baseFormat, type, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
		tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,layers,width,height,1 };
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTextureCubemap(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	auto tempHeight = height;

	constexpr GLenum cubeFaceTargets[] = {
		GL_TEXTURE_CUBE_MAP_POSITIVE_X,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
	};
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { if (!Bind()) { return false; } }
	for (auto& cubeFaceTarget : cubeFaceTargets) {
		tempWidth = width;
		tempHeight = height;
		for (GLint level = 0; level < levels; ++level) {
			glTexImage2D(cubeFaceTarget, level, internalFormat, tempWidth, tempHeight, 0, baseFormat, type, nullptr);
			tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
			tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		}
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,1 };
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTextureCubemapArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	auto tempHeight = height;

	constexpr GLenum cubeFaceTargets[] = {
		GL_TEXTURE_CUBE_MAP_POSITIVE_X,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
	};
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { if (!Bind()) { return false; } }
	for (auto& cubeFaceTarget : cubeFaceTargets) {
		tempWidth = width;
		tempHeight = height;
		for (GLint level = 0; level < levels; ++level) {
			glTexImage3D(cubeFaceTarget, level, internalFormat, tempWidth, tempHeight, layers, 0, baseFormat, type, nullptr);
			tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
			tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		}
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,1 };
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImage1DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLsizei width, GLint dstXOffset)
{
	GLsizei numBases  = GetGLNumBases(m_AllocationInfo->internalFormat,format,type);
	GLsizei pixelSize = numBases * GetGLTypeSize(type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	GLsizei mipWidth = GetMipWidth(level);
	GLsizei mipHeight = GetMipHeight(level);
	GLsizei mipDepth = GetMipDepth(level);
	if (mipWidth <= dstXOffset || mipWidth <= dstXOffset + width) {
		return false;
	}
	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		Bind();
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	glTexSubImage1D(m_Target, level, dstXOffset, width, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImage2DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLsizei width, GLsizei height, GLint dstXOffset, GLint dstYOffset)
{
	GLsizei numBases = GetGLNumBases(m_AllocationInfo->internalFormat, format, type);
	GLsizei pixelSize = numBases * GetGLTypeSize(type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	GLsizei mipWidth = GetMipWidth(level);
	GLsizei mipHeight = GetMipHeight(level);

	if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
		return false;
	}

	if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
		return false;
	}
	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		Bind();
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	glTexSubImage2D(m_Target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImage3DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset)
{
	GLsizei numBases = GetGLNumBases(m_AllocationInfo->internalFormat, format, type);
	GLsizei pixelSize = numBases * GetGLTypeSize(type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	GLsizei mipWidth = GetMipWidth(level);
	GLsizei mipHeight = GetMipHeight(level);
	GLsizei mipDepth = GetMipDepth(level);

	if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
		return false;
	}

	if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
		return false;
	}

	if (mipDepth <= dstZOffset || mipDepth < dstZOffset + depth) {
		return false;
	}
	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		Bind();
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	glTexSubImage3D(m_Target, level, dstXOffset, dstYOffset, dstZOffset, width, height, depth, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyLayeredImage1DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLint dstXOffset)
{
	GLsizei numBases = GetGLNumBases(m_AllocationInfo->internalFormat, format, type);
	GLsizei pixelSize = numBases * GetGLTypeSize(type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	GLsizei mipWidth = GetMipWidth(level);

	if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
		return false;
	}

	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		Bind();
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	glTexSubImage2D(m_Target, level, dstXOffset, layer, width, layers, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyLayeredImage2DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLint dstXOffset, GLint dstYOffset)
{
	GLsizei numBases = GetGLNumBases(m_AllocationInfo->internalFormat, format, type);
	GLsizei pixelSize = numBases * GetGLTypeSize(type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	GLsizei mipWidth = GetMipWidth(level);
	GLsizei mipHeight = GetMipHeight(level);

	if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
		return false;
	}

	if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
		return false;
	}
	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		Bind();
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	glTexSubImage3D(m_Target, level, dstXOffset, dstYOffset, layer, width, height, layers, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImage2DFromMemory(GLenum target, const void* pData, GLenum format, GLenum type, GLint level, GLsizei width, GLsizei height, GLint dstXOffset, GLint dstYOffset)
{
	GLsizei numBases = GetGLNumBases(m_AllocationInfo->internalFormat, format, type);
	GLsizei pixelSize = numBases * GetGLTypeSize(type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	GLsizei mipWidth = GetMipWidth(level);
	GLsizei mipHeight = GetMipHeight(level);

	if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
		return false;
	}

	if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
		return false;
	}
	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		if (!Bind()) { return false; }
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	glTexSubImage2D(target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyLayeredFaceImage2DFromMemory(GLenum target, const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLint dstXOffset, GLint dstYOffset)
{
	GLsizei numBases = GetGLNumBases(m_AllocationInfo->internalFormat, format, type);
	GLsizei pixelSize = numBases * GetGLTypeSize(type);
	GLsizei alignment = 0;
	if (pixelSize == 0) {
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}
	else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	GLsizei mipWidth = GetMipWidth(level);
	GLsizei mipHeight = GetMipHeight(level);

	if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
		return false;
	}

	if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
		return false;
	}
	bool bindedForCopy = !IsBinded();
	if (bindedForCopy) {
		if (!Bind()) { return false; }
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	glTexSubImage3D(target, level, dstXOffset, dstYOffset, layer, width, height, layers, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

