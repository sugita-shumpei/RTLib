#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include "GLContextImpl.h"
#include "GLTypeConversions.h"
#include <cassert>
RTLib::Ext::GL::GLContext:: GLContext() noexcept {
	m_Impl = std::unique_ptr<Impl>(new Impl());
}

RTLib::Ext::GL::GLContext::~GLContext() noexcept {
	m_Impl.reset();
}

bool RTLib::Ext::GL::GLContext::Initialize()
{
	if (!m_Impl) {
		return false;
	}
	if (m_Impl->m_IsInitialized) {
		return false;
	}
	if (!InitLoader()) {
		return false;
	}
	glGetIntegerv(GL_MAJOR_VERSION              , &m_Impl->m_GLMajorVersion);
	glGetIntegerv(GL_MINOR_VERSION              , &m_Impl->m_GLMinorVersion);
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK       , &m_Impl->m_GLProfileMask);
	glGetIntegerv(GL_MAX_IMAGE_UNITS            , &m_Impl->m_GLMaxImageUnits);
	m_Impl->m_Textures = std::vector<GLTexture*>(m_Impl->m_GLMaxImageUnits, nullptr);
	m_Impl->m_IsInitialized = true;
	return true;
}

void RTLib::Ext::GL::GLContext::Terminate()
{
	if (!m_Impl->m_IsInitialized) {
		return;
	}
	FreeLoader();
	m_Impl->m_IsInitialized = false;
}

bool RTLib::Ext::GL::GLContext::SupportVersion(uint32_t majorVersion, uint32_t minorVersion) const noexcept
{
	return m_Impl->SupportVersion(majorVersion, minorVersion);
}

auto RTLib::Ext::GL::GLContext::GetMajorVersion() const noexcept -> uint32_t
{
	return m_Impl->m_GLMajorVersion;
}

auto RTLib::Ext::GL::GLContext::GetMinorVersion() const noexcept -> uint32_t
{
	return m_Impl->m_GLMinorVersion;
}

bool RTLib::Ext::GL::GLContext::CopyBuffer(GLBuffer* srcBuffer, GLBuffer* dstBuffer, const std::vector<GLBufferCopy>& regions)
{
	if (!srcBuffer || !dstBuffer) { return false; }
	if (SupportVersion(4, 5)) {
		for (const auto& region : regions) {
			glCopyNamedBufferSubData(srcBuffer->GetResId(), dstBuffer->GetResId(), region.srcOffset, region.dstOffset, region.size);
		}
	}
	else {
		auto srcTarget = srcBuffer->GetCurrentTarget();
		auto dstTarget = dstBuffer->GetCurrentTarget();
		if (!srcTarget) {
			srcTarget  = GL_COPY_READ_BUFFER;
			glBindBuffer(GL_COPY_READ_BUFFER, srcBuffer->GetResId());
		}
		if (!dstTarget) {
			dstTarget = GL_COPY_WRITE_BUFFER;
			glBindBuffer(GL_COPY_WRITE_BUFFER, dstBuffer->GetResId());
		}
		for (const auto& region : regions) {
			glCopyBufferSubData(*srcTarget,*dstTarget, region.srcOffset, region.dstOffset, region.size);
		}
	}
	return true;
}

bool RTLib::Ext::GL::GLContext::CopyMemoryToBuffer(GLBuffer* buffer, const std::vector<GLMemoryBufferCopy>& regions)
{
	if (!buffer) { return false; }
	if (!(buffer->GetMemoryProperty() & GLMemoryPropertyHostWrite)) {
		return false;
	}
	char* mappedData = nullptr;
	if (SupportVersion(4, 5)) {
		for (const auto& region : regions) {
			mappedData = (char*)glMapNamedBufferRange(buffer->GetResId(), region.dstOffset, region.size, GL_MAP_WRITE_BIT);
			std::memcpy(mappedData, region.srcData, region.size);
			assert(glUnmapNamedBuffer(buffer->GetResId()));
		}
	}
	else {
		auto curTarget = buffer->GetCurrentTarget();
		if (!curTarget) {
			glBindBuffer(GL_COPY_READ_BUFFER, buffer->GetResId());
			curTarget =  GL_COPY_READ_BUFFER;
		}
		for (const auto& region : regions) {
			mappedData = (char*)glMapBufferRange(*curTarget, region.dstOffset, region.size, GL_MAP_WRITE_BIT);
			std::memcpy(mappedData, region.srcData, region.size);
			assert(glUnmapBuffer(*curTarget));
		}
	}
	return true;
}

bool RTLib::Ext::GL::GLContext::CopyBufferToMemory(GLBuffer* buffer, const std::vector<GLBufferMemoryCopy>& regions)
{
	if (!buffer) { return false; }
	if (!(buffer->GetMemoryProperty() & GLMemoryPropertyHostRead)) {
		return false;
	}
	char* mappedData = nullptr;
	if (SupportVersion(4, 5)) {
		for (const auto& region : regions) {
			mappedData = (char*)glMapNamedBufferRange(buffer->GetResId(), region.srcOffset, region.size, GL_MAP_READ_BIT);
			std::memcpy(region.dstData, mappedData, region.size);
			assert(glUnmapNamedBuffer(buffer->GetResId()));
		}
	}
	else {
		auto curTarget = buffer->GetCurrentTarget();
		if (!curTarget) {
			glBindBuffer(GL_COPY_READ_BUFFER, buffer->GetResId());
			curTarget = GL_COPY_READ_BUFFER;
		}
		for (const auto& region : regions) {
			mappedData = (char*)glMapBufferRange(*curTarget, region.srcOffset, region.size, GL_MAP_READ_BIT);
			std::memcpy(region.dstData, mappedData, region.size);
			assert(glUnmapBuffer(*curTarget));
		}
	}
	return true;
}

bool RTLib::Ext::GL::GLContext::CopyImageToBuffer(GLImage* srcImage, GLBuffer* dstBuffer, const std::vector<GLBufferImageCopy>& regions)
{

	return false;
}

bool RTLib::Ext::GL::GLContext::CopyBufferToImage(GLBuffer* srcBuffer, GLImage* dstImage, const std::vector<GLBufferImageCopy>& regions)
{
	if (!dstImage || !srcBuffer) { return false; }
	if (!(srcBuffer->GetBufferUsage() & GLBufferUsageImageCopySrc)) {
		return false;
	}
	auto viewType   = dstImage->GetViewType();
	auto format     = dstImage->GetFormat();
	auto baseFormat = GetGLFormatBaseFormat(format);
	auto baseEnum   = GetGLBaseFormatGLenum(baseFormat);
	auto typeEnum   = GetGLFormatGLUnpackEnum(format);
	auto type       = GetGLenumGLType(typeEnum);
	auto typeSize   = GetGLTypeSize(type);
	auto alignment  = static_cast<int>(typeSize);
	auto target     = GetGLImageViewTypeGLenum(viewType);
	auto resId      = dstImage->GetResId();
	if      ((typeSize % 8) == 0) { alignment = 8; }
	else if ((typeSize % 4) == 0) { alignment = 4; }
	else if ((typeSize % 2) == 0) { alignment = 2; }
	else                          { alignment = 1; }
	constexpr GLenum faceTargets[6] = {
		GL_TEXTURE_CUBE_MAP_POSITIVE_X,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	};
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, srcBuffer->GetResId());
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	if (SupportVersion(4, 5)) {
		switch (viewType)
		{
		case RTLib::Ext::GL::GLImageViewType::e1D:
			for (const auto& region : regions) {
				glTextureSubImage1D(resId, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageExtent.width, baseEnum, typeEnum, reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2D:
			for (const auto& region : regions) {
				glTextureSubImage2D(resId, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageExtent.width, region.imageExtent.height, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e3D:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageOffset.z, region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemap:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageSubresources.baseArrayLayer, region.imageExtent.width, region.imageExtent.height, region.imageSubresources.layerCount, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e1DArray:
			for (const auto& region : regions) {
				glTextureSubImage2D(resId, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageSubresources.baseArrayLayer, region.imageExtent.width, region.imageSubresources.layerCount, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2DArray:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageSubresources.baseArrayLayer, region.imageExtent.width, region.imageExtent.height, region.imageSubresources.layerCount, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemapArray:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageSubresources.baseArrayLayer, region.imageExtent.width, region.imageExtent.height, region.imageSubresources.layerCount, baseEnum, typeEnum, reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		default:
			break;
		}
	}
	else {
		GLint faceTargetIdxMin = 0;
		GLint faceTargetIdxMax = 0;
		GLint faceLayerIdxMin = 0;
		GLint faceLayerIdxMax = 0;
		glBindTexture(target, resId);
		switch (viewType)
		{
		case RTLib::Ext::GL::GLImageViewType::e1D:
			for (const auto& region : regions) {
				glTexSubImage1D(GL_TEXTURE_1D, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageExtent.width, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2D:
			for (const auto& region : regions) {
				glTexSubImage2D(GL_TEXTURE_2D, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageExtent.width, region.imageExtent.height, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e3D:
			for (const auto& region : regions) {
				glTexSubImage3D(GL_TEXTURE_3D, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageOffset.z, region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemap:
			for (const auto& region : regions) {
				faceTargetIdxMin = region.imageSubresources.baseArrayLayer % 6;
				faceTargetIdxMax = std::max<int>(faceTargetIdxMin + region.imageSubresources.layerCount, 6);
				for (int i = faceTargetIdxMin; i < faceTargetIdxMax; ++i) {
					glTexSubImage2D(faceTargets[i], region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageExtent.width, region.imageExtent.height, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
				}
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e1DArray:
			for (const auto& region : regions) {
				glTexSubImage2D(GL_TEXTURE_1D_ARRAY, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageSubresources.baseArrayLayer, region.imageExtent.width, region.imageSubresources.layerCount, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2DArray:
			for (const auto& region : regions) {
				glTexSubImage3D(GL_TEXTURE_2D_ARRAY, region.imageSubresources.mipLevel, region.imageOffset.x, region.imageOffset.y, region.imageSubresources.baseArrayLayer, region.imageExtent.width, region.imageExtent.height, region.imageSubresources.layerCount, baseEnum, typeEnum,  reinterpret_cast<const void*>(region.bufferOffset));
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemapArray:
			//TODO
			break;
		default:
			break;
		}
	}
	return true;
}

bool RTLib::Ext::GL::GLContext::CopyImageToMemory(GLImage* image, const std::vector<GLImageMemoryCopy>& regions)
{
	//TODO
	
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyMemoryToImage(GLImage* image, const std::vector<GLMemoryImageCopy>& regions)
{
	if (!image) { return false; }
	auto viewType   = image->GetViewType();
	auto format     = image->GetFormat();
	auto baseFormat = GetGLFormatBaseFormat(format);
	auto baseEnum   = GetGLBaseFormatGLenum(baseFormat);
	auto type       = GetGLFormatGLUnpackEnum(format);
	auto typeSize   = GetGLTypeSize(GetGLenumGLType(type));
	auto alignment = static_cast<int>(typeSize);
	auto target     = GetGLImageViewTypeGLenum(viewType);
	auto resId      = image->GetResId();
	if ((typeSize % 8) == 0) { alignment = 8; }
	else if ((typeSize % 4) == 0) { alignment = 4; }
	else if ((typeSize % 2) == 0) { alignment = 2; }
	else { alignment = 1; }
    constexpr GLenum faceTargets[6] = {
		GL_TEXTURE_CUBE_MAP_POSITIVE_X,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
		GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
		GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	};
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	if (SupportVersion(4, 5)) {
		switch (viewType)
		{
		case RTLib::Ext::GL::GLImageViewType::e1D:
			for (const auto& region : regions) {
				glTextureSubImage1D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageExtent.width, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2D:
			for (const auto& region : regions) {
				glTextureSubImage2D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageExtent.width, region.dstImageExtent.height, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e3D:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageOffset.z, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageExtent.depth, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemap:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y,region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageExtent.height,region.dstImageSubresources.layerCount, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e1DArray:
			for (const auto& region : regions) {
				glTextureSubImage2D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageSubresources.layerCount, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2DArray:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageSubresources.layerCount, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemapArray:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageSubresources.layerCount, baseEnum, type, region.srcData);
			}
			break;
		default:
			break;
		}
	}
	else {
		GLint faceTargetIdxMin = 0;
		GLint faceTargetIdxMax = 0;
		GLint faceLayerIdxMin  = 0;
		GLint faceLayerIdxMax  = 0;
		glBindTexture(target,resId);
		switch (viewType)
		{
		case RTLib::Ext::GL::GLImageViewType::e1D:
			for (const auto& region : regions) {
				glTexSubImage1D(GL_TEXTURE_1D, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageExtent.width, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2D:
			for (const auto& region : regions) {
				glTexSubImage2D(GL_TEXTURE_2D, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageExtent.width, region.dstImageExtent.height, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e3D:
			for (const auto& region : regions) {
				glTexSubImage3D(GL_TEXTURE_3D, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageOffset.z, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageExtent.depth, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemap:
			for (const auto& region : regions) {
				faceTargetIdxMin = region.dstImageSubresources.baseArrayLayer % 6;
				faceTargetIdxMax = std::max<int>(faceTargetIdxMin + region.dstImageSubresources.layerCount, 6);
				for (int i = faceTargetIdxMin; i < faceTargetIdxMax; ++i) {
					glTexSubImage2D(faceTargets[i], region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageExtent.width, region.dstImageExtent.height, baseEnum, type, region.srcData);
				}
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e1DArray:
			for (const auto& region : regions) {
				glTexSubImage2D(GL_TEXTURE_1D_ARRAY, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageSubresources.layerCount, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2DArray:
			for (const auto& region : regions) {
				glTexSubImage3D(GL_TEXTURE_2D_ARRAY, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageSubresources.layerCount, baseEnum, type, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemapArray:
			//TODO
			break;
		default:
			break;
		}
	}
	return true;
}

void RTLib::Ext::GL::GLContext::SetUniformImageUnit(int loc, int index)
{
	if (!m_Impl->m_Program) { return; }
	if (SupportVersion(4, 1)) {
		m_Impl->m_Program->SetUniformImageUnit(loc, index);
	}
	else {
		glUniform1i(loc, index);
		m_Impl->m_Program->SetImageUnit(loc, index);
	}
}

void RTLib::Ext::GL::GLContext::SetProgramUniformImageUnit(GLProgram* program, int loc, int index)
{
}

void RTLib::Ext::GL::GLContext::SetActiveTexture(int index)
{
	if (index < 0) {
		index += m_Impl->m_GLMaxImageUnits;
	}
	assert((index >= 0) && (index < m_Impl->m_GLMaxImageUnits));
	if (index != m_Impl->m_ActiveTexUnit) {
		glActiveTexture(GL_TEXTURE0 + index);
		m_Impl->m_ActiveTexUnit = index;
	}
}

void RTLib::Ext::GL::GLContext::SetTexture(int index, GLTexture* texture)
{
	if (!texture) {
		return;
	}
	if (index < 0) {
		index += m_Impl->m_GLMaxImageUnits;
	}
	assert((index >= 0) && (index < m_Impl->m_GLMaxImageUnits));
	if (m_Impl->m_ActiveTexUnit != index) {
		SetActiveTexture(index);
	}
	if (m_Impl->m_Textures[index] != texture) {
		auto viewType = texture->GetType();
		auto target = GetGLImageViewTypeGLenum(viewType);
		glBindTexture(target, texture->GetResId());
		m_Impl->m_Textures[index] = texture;
	}
}

void RTLib::Ext::GL::GLContext::DrawArrays(GLDrawMode drawMode, size_t first, int32_t count)
{
    if (!m_Impl->m_VAO){ return;}
	glDrawArrays(GetGLDrawModeGLenum(drawMode), first, count);
}

void RTLib::Ext::GL::GLContext::DrawElements(GLDrawMode drawMode, GLIndexFormat indexType, size_t count, intptr_t indexOffsetInBytes)
{
    if (!m_Impl->m_VAO){ return;}
	glDrawElements(GetGLDrawModeGLenum(drawMode), count, GetGLTypeGLEnum(static_cast<GLTypeFlagBits>(indexType)), reinterpret_cast<void*>(indexOffsetInBytes));
}

void RTLib::Ext::GL::GLContext::SetClearBuffer(GLClearBufferFlags flags)
{
	GLbitfield mask = 0;
	if ((flags & GLClearBufferFlagsColor)) {
		mask |= GL_COLOR_BUFFER_BIT;
	}
	if ((flags & GLClearBufferFlagsDepth)) {
		mask |= GL_DEPTH_BUFFER_BIT;
	}
	if ((flags & GLClearBufferFlagsStencil)) {
		mask |= GL_STENCIL_BUFFER_BIT;
	}
	glClear(mask);
}

void RTLib::Ext::GL::GLContext::SetClearColor(float r, float g, float b, float a)
{
	glClearColor(r, g, b, a);
}

auto RTLib::Ext::GL::GLContext::CreateBuffer(const GLBufferCreateDesc& desc) -> GLBuffer*
{
	return GLBuffer::Allocate(this,desc);
}

auto RTLib::Ext::GL::GLContext::CreateImage(const GLImageCreateDesc& desc) -> GLImage*
{
	return GLImage::Allocate(this,desc);
}

auto RTLib::Ext::GL::GLContext::CreateTexture(const GLTextureCreateDesc& desc) -> GLTexture*
{
	return GLTexture::Allocate(this,desc);
}

auto RTLib::Ext::GL::GLContext::CreateShader(GLShaderStageFlagBits shaderType) -> GLShader*
{
	return GLShader::New(this,shaderType,this->SupportVersion(4,6));
}

auto RTLib::Ext::GL::GLContext::CreateProgram() -> GLProgram*
{
	return GLProgram::New(this);
}

auto RTLib::Ext::GL::GLContext::CreateVertexArray()->GLVertexArray*{
    return GLVertexArray::New(this);
}

void RTLib::Ext::GL::GLContext::SetProgram(GLProgram* program){
    if (!program){ return; }
    if (m_Impl->m_Program == program){
        return;
    }
    m_Impl->m_Program = program;
    m_Impl->m_Program->Enable();
}

void RTLib::Ext::GL::GLContext::SetVertexArrayState(GLVertexArray* vao)
{
    if (!vao){ return;}
    if (m_Impl->m_VAO == vao){
        return;
    }
    m_Impl->m_VAO = vao;
    m_Impl->m_VAO->Bind();
}
