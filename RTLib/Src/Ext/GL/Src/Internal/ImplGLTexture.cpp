#include "ImplGLTexture.h"
#include "ImplGLBuffer.h"
#include "ImplGLUtils.h"
bool RTLib::Ext::GL::Internal::ImplGLTexture::Allocate(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth)
{
	if (IsAllocated()|| width == 0|| height == 0|| depth == 0) {
		return false;
	}
	switch (m_Target) {
	case GL_TEXTURE_1D:
		if (layers !=1||height != 1 || depth != 1) {
			return false;
		}
		AllocateTexture1D(internalFormat, levels, width);
		return true;
	case GL_TEXTURE_2D:
		if (layers != 1 || depth != 1) {
			return false;
		}
		AllocateTexture2D(internalFormat, levels, width, height);
		return true;
	case GL_TEXTURE_3D:
		if (layers != 1) {
			return false;
		}
		AllocateTexture3D(internalFormat, levels, width, height, depth);
		return true;
	case GL_TEXTURE_1D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		AllocateTexture1DArray(internalFormat, levels, layers, width);
		return true;

	case GL_TEXTURE_2D_ARRAY:
		if (depth != 1) {
			return false;
		}
		AllocateTexture2DArray(internalFormat, levels, layers, width, height);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layers != 1 || depth != 1 || levels != 1) {
			return false;
		}
		AllocateTexture2D(internalFormat, 1, width, height);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layers != 1 || depth != 1) {
			return false;
		}
		AllocateTexture2D(internalFormat, levels, width, height);
		return true;

	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		if (depth != 1) {
			return false;
		}
		AllocateTexture2DArray(internalFormat, levels, layers, width, height);
		return true;
	case GL_TEXTURE_CUBE_MAP:
		if (layers != 1 || depth != 1) {
			return false;
		}
		AllocateTextureCubemap(internalFormat, levels, width, height);
		return true;

	case GL_TEXTURE_CUBE_MAP_ARRAY:
		if (depth != 1) {
			return false;
		}
		AllocateTextureCubemapArray(internalFormat, levels,layers, width, height);
		return true;
	default:
		return false;
	}
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImageFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset)
{
	if (!pData || !IsBinded() || !IsAllocated() || width == 0 || height == 0 || depth == 0 || level < 0 || !m_BPBuffer) {
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
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImageToMemory(void* pData, GLenum format, GLenum type, GLint level)
{
	if (!pData || !IsBinded() || !IsAllocated() || level < 0 || !m_BPBuffer) {
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
	glPixelStorei(GL_PACK_ALIGNMENT, alignment);
	glGetTexImage(m_Target, level, format, type, pData);
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImageFromBuffer(ImplGLBuffer* src, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset, GLintptr srcOffset)
{
	if (!IsBinded() || !IsAllocated() || width == 0 || height == 0 || depth == 0 || level < 0 || !m_BPBuffer || !src) {
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
		if (bindForCopy) { src->Bind(GL_PIXEL_UNPACK_BUFFER); }
		res = CopyImage1DFromMemory(pData, format, type, level, width, dstXOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_2D:
		if (layer != 0 || layers != 1 || depth != 1) {

			return false;
		}
		if (bindForCopy) { src->Bind(GL_PIXEL_UNPACK_BUFFER); }
		res = CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_3D:
		if (layer != 0 || layers != 1) {
			return false;
		}
		if (bindForCopy) { src->Bind(GL_PIXEL_UNPACK_BUFFER); }
		res = CopyImage3DFromMemory(pData, format, type, level, width, height, depth, dstXOffset, dstYOffset, dstZOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_1D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) { src->Bind(GL_PIXEL_UNPACK_BUFFER); }
		res = CopyLayeredImage1DFromMemory(pData, format, type, level, layer, layers, width, dstXOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_2D_ARRAY:
		if (depth != 1) {

			return false;
		}
		if (bindForCopy) { src->Bind(GL_PIXEL_UNPACK_BUFFER); }
		res = CopyLayeredImage2DFromMemory(pData, format, type, level, layer, layers, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) { src->Bind(GL_PIXEL_UNPACK_BUFFER); }
		res = CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) { src->Bind(GL_PIXEL_UNPACK_BUFFER); }
		res = CopyImage2DFromMemory(pData, format, type, level, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (bindForCopy) { src->Bind(GL_PIXEL_UNPACK_BUFFER); }
		res = CopyLayeredImage2DFromMemory(pData, format, type, level, layer, layers, width, height, dstXOffset, dstYOffset);
		if (bindForCopy) { src->Unbind(); }
		return res;
	default:
		return false;
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImageToBuffer(ImplGLBuffer* dst, GLenum format, GLenum type, GLint level, GLintptr dstOffset)
{
	if (!dst || !IsBinded() || !IsAllocated() || level < 0) {

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
	bool bindedForCopySrc = false;
	GLenum srcTarget;
	if (!dst->IsBinded()||!dst->IsAllocated()) {
		return false;
	}
	else{
		if (dst->GetTarget() != GL_PIXEL_PACK_BUFFER) {
			return false;
		}
	}
	glPixelStorei(GL_PACK_ALIGNMENT, alignment);
	glGetTexImage(m_Target, level, format, type, reinterpret_cast<void*>(dstOffset));
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImageFromMemory(GLenum target, const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset)
{
	if (!pData || !IsBinded() || !IsAllocated() || width == 0 || height == 0 || depth == 0 || level < 0 || !m_BPBuffer) {
		return false;
	}
	if (m_BPBuffer->HasBindable(GL_PIXEL_UNPACK_BUFFER)) {
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
	GLsizei mipWidth = GetMipWidth(level);
	GLsizei mipHeight = GetMipHeight(level);
	if (layer != 0 || layers != 1 || depth != 1) {
		return false;
	}

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
	glTexSubImage2D(target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
	if (bindedForCopy) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImageToMemory(GLenum target, void* pData, GLenum format, GLenum type, GLint level)
{
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImageFromBuffer(GLenum target, ImplGLBuffer* src, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset, GLintptr srcOffset)
{
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFaceImageToBuffer(GLenum target, ImplGLBuffer* src, GLenum format, GLenum type, GLint level, GLintptr srcOffset)
{
	return false;
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

void RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture1D(GLenum internalFormat, GLint levels, GLsizei width)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { Bind(); }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage1D(m_Target, level, internalFormat, tempWidth, 0, baseFormat, GL_UNSIGNED_BYTE, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,1,1 };
}

void RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture2D(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	auto tempHeight = height;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { Bind(); }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage2D(m_Target, level, internalFormat, tempWidth, tempHeight, 0, baseFormat, type, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
		tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,1 };
}

void RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture3D(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height, GLsizei depth)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	auto tempHeight = height;
	auto tempDepth = depth;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { Bind(); }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage3D(m_Target, level, internalFormat, tempWidth, tempHeight, tempDepth, 0, baseFormat, type, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
		tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		tempDepth = std::max<GLsizei>(tempDepth / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,depth };
}

void RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture1DArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { Bind(); }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage2D(m_Target, level, internalFormat, tempWidth, layers, 0, baseFormat, type, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,layers,width,1,1 };
}

void RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTexture2DArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height)
{
	auto baseFormat = GetGLBaseFormat(internalFormat);
	auto type = GetGLBaseType(internalFormat);
	auto tempWidth = width;
	auto tempHeight = height;
	bool copyForExecuted = !IsBinded();
	if (copyForExecuted) { Bind(); }
	for (GLint level = 0; level < levels; ++level) {
		glTexImage3D(m_Target, level, internalFormat, tempWidth, tempHeight, layers, 0, baseFormat, type, nullptr);
		tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
		tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
	}
	if (copyForExecuted) { Unbind(); }
	m_AllocationInfo = AllocationInfo{ levels,internalFormat,layers,width,height,1 };
}

void RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTextureCubemap(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height)
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
	if (copyForExecuted) { Bind(); }
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
}

void RTLib::Ext::GL::Internal::ImplGLTexture::AllocateTextureCubemapArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height)
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
	if (copyForExecuted) { Bind(); }
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
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyImage1DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLsizei width, GLint dstXOffset)
{
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

