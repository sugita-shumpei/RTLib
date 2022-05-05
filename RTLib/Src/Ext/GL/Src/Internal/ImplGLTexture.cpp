#include "ImplGLTexture.h"
#include "ImplGLBuffer.h"
bool RTLib::Ext::GL::Internal::ImplGLTexture::Allocate(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth)
{
	if (!IsBinded()||IsAllocated()|| width == 0|| height == 0|| depth == 0) {
		return false;
	}
	auto baseFormat = ConvertInternalFormatToBase(internalFormat);
	auto type       = QuerySuitableTypeFromFormat(internalFormat);
	auto tempWidth  = width;
	auto tempHeight = height;
	auto tempDepth  = depth;

	std::vector<GLenum> cubeFaceTargets = {
		GL_TEXTURE_CUBE_MAP_POSITIVE_X,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
	};

	switch (m_Target) {
	case GL_TEXTURE_1D:
		if (layers !=1||height != 1 || depth != 1) {
			return false;
		}
		for (GLint level = 0; level <levels;++level) {
			glTexImage1D(m_Target, level, internalFormat, tempWidth, 0, baseFormat, GL_UNSIGNED_BYTE, nullptr);
			tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
		}
		m_AllocationInfo = AllocationInfo{levels,internalFormat,1,width,1,1};
		return true;
	case GL_TEXTURE_2D:
		if (layers != 1 || depth != 1) {
			return false;
		}
		for (GLint level = 0; level < levels; ++level) {
			glTexImage2D(m_Target, level, internalFormat, tempWidth, tempHeight, 0, baseFormat, type, nullptr);
			tempWidth  = std::max<GLsizei>(tempWidth / 2, 1);
			tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,1 };
		return true;
	case GL_TEXTURE_3D:
		if (layers != 1) {
			return false;
		}
		for (GLint level = 0; level < levels; ++level) {
			glTexImage3D(m_Target, level, internalFormat, tempWidth, tempHeight, tempDepth, 0, baseFormat, type, nullptr);
			tempWidth  = std::max<GLsizei>(tempWidth  / 2, 1);
			tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
			tempDepth  = std::max<GLsizei>(tempDepth  / 2, 1);
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,depth };
		return true;
	case GL_TEXTURE_1D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		for (GLint level = 0; level < levels; ++level) {
			glTexImage2D(m_Target, level, internalFormat, tempWidth,layers, 0, baseFormat, type, nullptr);
			tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,layers,width,1,1 };
		return true;

	case GL_TEXTURE_2D_ARRAY:
		if (depth != 1) {
			return false;
		}
		for (GLint level = 0; level < levels; ++level) {
			glTexImage3D(m_Target, level, internalFormat, tempWidth, tempHeight, layers, 0, baseFormat, type, nullptr);
			tempWidth  = std::max<GLsizei>(tempWidth / 2, 1);
			tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,layers,width,height,1 };
		return true;

	case GL_TEXTURE_RECTANGLE:
		if (layers != 1 || depth != 1) {
			return false;
		}
		for (GLint level = 0; level < levels; ++level) {
			glTexImage2D(m_Target, level, internalFormat, tempWidth, tempHeight, 0, baseFormat, type, nullptr);
			tempWidth  = std::max<GLsizei>(tempWidth  / 2, 1);
			tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,1 };
		return true;

	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layers != 1 || depth != 1) {
			return false;
		}
		for (GLint level = 0; level < levels; ++level) {
			glTexImage2D(m_Target, level, internalFormat, tempWidth, tempHeight, 0, baseFormat, type, nullptr);
			tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
			tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,1 };
		return true;

	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		if (depth != 1) {
			return false;
		}
		for (GLint level = 0; level < levels; ++level) {
			glTexImage3D(m_Target, level, internalFormat, tempWidth, tempHeight, layers, 0, baseFormat, type, nullptr);
			tempWidth  = std::max<GLsizei>(tempWidth / 2, 1);
			tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,layers,width,height,1 };
		return true;
	case GL_TEXTURE_CUBE_MAP:
		if (layers != 1 || depth != 1) {
			return false;
		}
		for (auto& cubeFaceTarget : cubeFaceTargets) {
			tempWidth  = width;
			tempHeight = height;
			for (GLint level = 0; level < levels; ++level) {
				glTexImage2D(cubeFaceTarget, level, internalFormat, tempWidth, tempHeight, 0, baseFormat, type, nullptr);
				tempWidth  = std::max<GLsizei>(tempWidth / 2, 1);
				tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
			}
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,1,width,height,1 };
		return true;

	case GL_TEXTURE_CUBE_MAP_ARRAY:
		if (depth != 1) {
			return false;
		}
		for (auto& cubeFaceTarget : cubeFaceTargets) {
			tempWidth = width;
			tempHeight = height;
			for (GLint level = 0; level < levels; ++level) {
				glTexImage3D(cubeFaceTarget, level, internalFormat, tempWidth, tempHeight, layers,0, baseFormat, type, nullptr);
				tempWidth = std::max<GLsizei>(tempWidth / 2, 1);
				tempHeight = std::max<GLsizei>(tempHeight / 2, 1);
			}
		}
		m_AllocationInfo = AllocationInfo{ levels,internalFormat,layers,width,height,1 };
		return true;
	default:
		return false;
	}
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset)
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
	GLsizei pixelSize = CalculatePixelSize(format, type);
	GLsizei alignment = 0;
	if (pixelSize == 0){
		return false;
	}
	if (pixelSize % 8 == 0) {
		alignment = 8;
	}else if (pixelSize % 4 == 0) {
		alignment = 4;
	}
	else if (pixelSize % 2 == 0) {
		alignment = 2;
	}
	else {
		alignment = 1;
	}
	GLsizei mipWidth  = GetMipWidth (level);
	GLsizei mipHeight = GetMipHeight(level);
	GLsizei mipDepth  = GetMipDepth (level);
	switch (m_Target) {
	case GL_TEXTURE_1D:
		if (layer != 0 || layers!=1 || height!=1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth <= dstXOffset + width) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage1D(m_Target, level, dstXOffset, width, format, type, pData);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0 || layers != 1 || depth != 1) {

			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {

			return false;
		}
		if (mipHeight<= dstYOffset || mipHeight< dstYOffset + height) {

			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage2D(m_Target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
		return true;
	case GL_TEXTURE_3D:
		if (layer != 0 || layers != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth  < dstXOffset + width  ) {
			return false;
		}
		if (mipHeight<= dstYOffset || mipHeight < dstYOffset + height ) {
			return false;
		}
		if (mipDepth <= dstZOffset || mipDepth < dstZOffset + depth  ) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage3D(m_Target, level, dstXOffset, dstYOffset, dstZOffset, width, height, depth, format, type, pData);
		return true;
	case GL_TEXTURE_1D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage2D(m_Target, level, dstXOffset, layer, width, layers, format, type, pData);
		return true;
	case GL_TEXTURE_2D_ARRAY:
		if (depth != 1) {

			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width ) {

			return false;
		}
		if (mipHeight<= dstYOffset || mipHeight< dstYOffset + height) {

			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage3D(m_Target, level, dstXOffset, dstYOffset, layer, width, height, layers, format, type, pData);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		if (mipHeight<= dstYOffset || mipHeight< dstYOffset + height) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage2D(m_Target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		if (mipHeight<= dstYOffset || mipHeight< dstYOffset + height) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage2D(m_Target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		if (mipHeight<= dstYOffset || mipHeight< dstYOffset + height) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage3D(m_Target, level, dstXOffset, dstYOffset, layer, width, height, layers, format, type, pData);
		return true;
	default:
		return false;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyToMemory(void* pData, GLenum format, GLenum type, GLint level)
{
	if (!pData || !IsBinded() || !IsAllocated() || level < 0 || !m_BPBuffer) {
		return false;
	}
	if (m_BPBuffer->HasBindable(GL_PIXEL_PACK_BUFFER)) {
		return false;
	}
	GLsizei pixelSize = CalculatePixelSize(format, type);
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

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyFromBuffer(ImplGLBuffer* src, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset, GLintptr srcOffset)
{
	if (!src || !IsBinded() || !IsAllocated() || width == 0 || height == 0 || depth == 0 || level < 0) {
		return false;
	}
	if (!src || !IsBinded() || !IsAllocated() || level < 0) {
		return false;
	}
	bool bindedForCopySrc = false;
	GLenum srcTarget;
	if (!src->IsBinded() || !src->IsAllocated()) {
		return false;
	}
	else {
		if (src->GetTarget() != GL_PIXEL_UNPACK_BUFFER) {
			return false;
		}
	}
	GLsizei pixelSize = CalculatePixelSize(format, type);
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
	const void* pData = reinterpret_cast<void*>(srcOffset);
	switch (m_Target) {
	case GL_TEXTURE_1D:
		if (layer != 0 || layers != 1 || height != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth <= dstXOffset + width) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage1D(m_Target, level, dstXOffset, width, format, type, pData);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		if (mipHeight<= dstYOffset || mipHeight< dstYOffset + height) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage2D(m_Target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
		return true;
	case GL_TEXTURE_3D:
		if (layer != 0 || layers != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
			return false;
		}
		if (mipDepth  <= dstZOffset || mipDepth < dstZOffset + depth) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage3D(m_Target, level, dstXOffset, dstYOffset, dstZOffset, width, height, depth, format, type, pData);
		return true;
	case GL_TEXTURE_1D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage2D(m_Target, level, dstXOffset, layers, width, layer, format, type, pData);
		return true;
	case GL_TEXTURE_2D_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage3D(m_Target, level, dstXOffset, dstYOffset, layers, width, height, layer, format, type, pData);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset  || mipWidth  < dstXOffset + width) {
			return false;
		}
		if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage2D(m_Target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || layers != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset  || mipWidth  < dstXOffset + width) {
			return false;
		}
		if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage2D(m_Target, level, dstXOffset, dstYOffset, width, height, format, type, pData);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		if (height != 1 || depth != 1) {
			return false;
		}
		if (mipWidth <= dstXOffset || mipWidth < dstXOffset + width) {
			return false;
		}
		if (mipHeight <= dstYOffset || mipHeight < dstYOffset + height) {
			return false;
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		glTexSubImage3D(m_Target, level, dstXOffset, dstYOffset, layers, width, height, layer, format, type, pData);
		return true;
	default:
		return false;
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLTexture::CopyToBuffer(ImplGLBuffer* dst, GLenum format, GLenum type, GLint level, GLintptr dstOffset)
{
	if (!dst || !IsBinded() || !IsAllocated() || level < 0) {

		return false;
	}
	GLsizei pixelSize = CalculatePixelSize(format, type);
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
