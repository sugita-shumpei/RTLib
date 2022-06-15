#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLRectRenderer.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <RTLib/Ext/GL/GLNatives.h>
#include <RTLib/Core/Exceptions.h>
#include <RTLib/Core/Utility.h>
#include "GLContextImpl.h"
#include "GLTypeConversions.h"
#include <vector>
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
	m_Impl->m_Images         = std::vector<GLImage*>(m_Impl->m_GLMaxImageUnits, nullptr);
	m_Impl->m_IsInitialized  = true;
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
			glCopyNamedBufferSubData(GLNatives::GetResId(srcBuffer), GLNatives::GetResId(dstBuffer), region.srcOffset, region.dstOffset, region.size);
		}
	}
	else {
		auto srcUsage = GLBufferUsageGenericCopySrc;
		auto dstUsage = GLBufferUsageGenericCopyDst;
		auto srcMainUsage = srcBuffer->GetMainUsage();
		auto dstMainUsage = dstBuffer->GetMainUsage();
		if ( srcMainUsage!= GLBufferUsageGenericCopyDst) {
			if (!GetBuffer(srcMainUsage)) {
				srcUsage = srcMainUsage;
			}
		}
		if ( dstMainUsage!= GLBufferUsageGenericCopySrc) {
			if (!GetBuffer(dstMainUsage)) {
				dstUsage = dstMainUsage;
			}
		}
		SetBuffer(srcUsage, srcBuffer);
		SetBuffer(dstUsage, dstBuffer);
		auto srcTarget = GetGLBufferMainUsageTarget(srcUsage);
		auto dstTarget = GetGLBufferMainUsageTarget(dstUsage);
		for (const auto& region : regions) {
			glCopyBufferSubData(srcTarget,dstTarget, region.srcOffset, region.dstOffset, region.size);
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
			RTLIB_CORE_ASSERT_IF_FAILED(glUnmapNamedBuffer(buffer->GetResId()));
		}
	}
	else {
		auto dstUsage = GLBufferUsageGenericCopyDst;
		auto dstMainUsage = buffer->GetMainUsage();
		if ( dstMainUsage!= GLBufferUsageGenericCopySrc) {
			if (!GetBuffer(dstMainUsage)) {
				dstUsage = dstMainUsage;
			}
		}
		SetBuffer(dstUsage, buffer);
		auto dstTarget = GetGLBufferMainUsageTarget(dstUsage);
		for (const auto& region : regions) {
			mappedData = (char*)glMapBufferRange(dstTarget, region.dstOffset, region.size, GL_MAP_WRITE_BIT);
			std::memcpy(mappedData, region.srcData, region.size);
			RTLIB_CORE_ASSERT_IF_FAILED(glUnmapBuffer(dstTarget));
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
			RTLIB_CORE_ASSERT_IF_FAILED(glUnmapNamedBuffer(buffer->GetResId()));
		}
	}
	else {
		auto srcUsage = GLBufferUsageGenericCopySrc;
		auto srcMainUsage = buffer->GetMainUsage();
		if (srcMainUsage != GLBufferUsageGenericCopyDst) {
			if (!GetBuffer(srcMainUsage)) {
				srcUsage = srcMainUsage;
			}
		}
		SetBuffer(srcUsage, buffer);
		auto srcTarget = GetGLBufferMainUsageTarget(srcUsage);
		for (const auto& region : regions) {
			mappedData = (char*)glMapBufferRange(srcTarget, region.srcOffset, region.size, GL_MAP_READ_BIT);
			std::memcpy(region.dstData, mappedData, region.size);
			RTLIB_CORE_ASSERT_IF_FAILED(glUnmapBuffer(srcTarget));
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
	if (!(srcBuffer->GetUsages() & GLBufferUsageImageCopySrc)) {
		return false;
	}
	auto viewType   = dstImage->GetViewType();
	auto format     = dstImage->GetFormat();
	auto baseFormat = GLFormatUtils::GetBaseFormat(format);
	auto baseEnum   = GetGLBaseFormatGLenum(baseFormat);
	auto typeEnum   = GetGLFormatGLUnpackEnum(format);
	auto type       = GetGLenumGLType(typeEnum);
	auto typeSize   = GLDataTypeFlagsUtils::GetBaseTypeBitSize(type)/8;
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
	SetBuffer(GLBufferUsageImageCopySrc, srcBuffer);
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
		SetImage(0, dstImage);
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
	auto baseFormat = GLFormatUtils::GetBaseFormat(format);
	auto baseEnum   = GetGLBaseFormatGLenum(baseFormat);
	auto typeEnum   = GetGLFormatGLUnpackEnum(format);
	auto type       = GetGLenumGLType(typeEnum);
	auto typeSize   = GLDataTypeFlagsUtils::GetBaseTypeBitSize(type) / 8;
	auto alignment   = static_cast<int>(typeSize);
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
	InvalidateBuffer(GLBufferUsageImageCopySrc);
	glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
	if (SupportVersion(4, 5)) {
		switch (viewType)
		{
		case RTLib::Ext::GL::GLImageViewType::e1D:
			for (const auto& region : regions) {
				glTextureSubImage1D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageExtent.width, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2D:
			for (const auto& region : regions) {
				glTextureSubImage2D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageExtent.width, region.dstImageExtent.height, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e3D:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageOffset.z, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageExtent.depth, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemap:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y,region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageExtent.height,region.dstImageSubresources.layerCount, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e1DArray:
			for (const auto& region : regions) {
				glTextureSubImage2D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageSubresources.layerCount, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2DArray:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageSubresources.layerCount, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemapArray:
			for (const auto& region : regions) {
				glTextureSubImage3D(resId, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageSubresources.layerCount, baseEnum, typeEnum, region.srcData);
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
		SetImage(0, image);
		switch (viewType)
		{
		case RTLib::Ext::GL::GLImageViewType::e1D:
			for (const auto& region : regions) {
				glTexSubImage1D(GL_TEXTURE_1D, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageExtent.width, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2D:
			for (const auto& region : regions) {
				glTexSubImage2D(GL_TEXTURE_2D, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageExtent.width, region.dstImageExtent.height, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e3D:
			for (const auto& region : regions) {
				glTexSubImage3D(GL_TEXTURE_3D, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageOffset.z, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageExtent.depth, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::eCubemap:
			for (const auto& region : regions) {
				faceTargetIdxMin = region.dstImageSubresources.baseArrayLayer % 6;
				faceTargetIdxMax = std::max<int>(faceTargetIdxMin + region.dstImageSubresources.layerCount, 6);
				for (int i = faceTargetIdxMin; i < faceTargetIdxMax; ++i) {
					glTexSubImage2D(faceTargets[i], region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageExtent.width, region.dstImageExtent.height, baseEnum, typeEnum, region.srcData);
				}
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e1DArray:
			for (const auto& region : regions) {
				glTexSubImage2D(GL_TEXTURE_1D_ARRAY, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageSubresources.layerCount, baseEnum, typeEnum, region.srcData);
			}
			break;
		case RTLib::Ext::GL::GLImageViewType::e2DArray:
			for (const auto& region : regions) {
				glTexSubImage3D(GL_TEXTURE_2D_ARRAY, region.dstImageSubresources.mipLevel, region.dstImageOffset.x, region.dstImageOffset.y, region.dstImageSubresources.baseArrayLayer, region.dstImageExtent.width, region.dstImageExtent.height, region.dstImageSubresources.layerCount, baseEnum, typeEnum, region.srcData);
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


/*Buffer*/

void RTLib::Ext::GL::GLContext::SetBuffer(GLBufferUsageFlagBits usage, GLBuffer* buffer, bool resetVAO)
{
	if (!buffer) { return; }
	if (!HasBuffer(usage, buffer) ||
		(resetVAO && ((usage == GLBufferUsageVertex) || (usage == GLBufferUsageIndex)))) {
		glBindBuffer(GetGLBufferMainUsageTarget(usage), buffer->GetResId());
		m_Impl->m_Buffers[RTLib::Core::Utility::Log2(static_cast<uint64_t>(usage))] = buffer;
	}
}

auto RTLib::Ext::GL::GLContext::GetBuffer(GLBufferUsageFlagBits usage) noexcept -> GLBuffer*
{
	return m_Impl->m_Buffers[RTLib::Core::Utility::Log2(static_cast<uint64_t>(usage))];
}

bool RTLib::Ext::GL::GLContext::HasBuffer(GLBufferUsageFlagBits usage, GLBuffer* buffer) const noexcept
{
	if (!buffer) { return false; }
	return m_Impl->m_Buffers[RTLib::Core::Utility::Log2(static_cast<uint64_t>(usage))] == buffer;
}

void RTLib::Ext::GL::GLContext::InvalidateBuffer(GLBufferUsageFlagBits usage)
{
	auto usageIndex = RTLib::Core::Utility::Log2(static_cast<uint64_t>(usage));
	if (m_Impl->m_Buffers[usageIndex]) {
		glBindBuffer(GetGLBufferMainUsageTarget(usage), 0);
		m_Impl->m_Buffers[usageIndex] = nullptr;
	}

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

void RTLib::Ext::GL::GLContext::ActivateImageUnit(int index)
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
	SetImage(index, texture->GetImage());
}

void RTLib::Ext::GL::GLContext::SetImage(int index, GLImage* image)
{
	if (!image) {
		return;
	}
	if (index < 0) {
		index += m_Impl->m_GLMaxImageUnits;
	}
	assert((index >= 0) && (index < m_Impl->m_GLMaxImageUnits));
	if (m_Impl->m_ActiveTexUnit != index) {
		ActivateImageUnit(index);
	}
	if (m_Impl->m_Images[index] != image) {
		auto viewType = image->GetViewType();
		auto target = GetGLImageViewTypeGLenum(viewType);
		glBindTexture(target, image->GetResId());
		m_Impl->m_Images[index] = image;
	}
}

void RTLib::Ext::GL::GLContext::DrawArrays(GLDrawMode drawMode, size_t first, int32_t count)
{
    if (!m_Impl->m_VertexArray){ return;}
	glDrawArrays(GetGLDrawModeGLenum(drawMode), first, count);
}

void RTLib::Ext::GL::GLContext::DrawElements(GLDrawMode drawMode, GLIndexFormat indexType, size_t count, intptr_t indexOffsetInBytes)
{
    if (!m_Impl->m_VertexArray){ return;}
	glDrawElements(GetGLDrawModeGLenum(drawMode), count, GetGLTypeGLEnum(static_cast<GLDataTypeFlagBits>(indexType)), reinterpret_cast<void*>(indexOffsetInBytes));
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

auto RTLib::Ext::GL::GLContext::CreateProgramPipeline() -> GLProgramPipeline*
{
	return GLProgramPipeline::New(this);
}

auto RTLib::Ext::GL::GLContext::CreateVertexArray()->GLVertexArray*{
    return GLVertexArray::New(this);
}

auto RTLib::Ext::GL::GLContext::CreateRectRenderer(const GLVertexArrayCreateDesc& desc) -> GLRectRenderer*
{
	return GLRectRenderer::New(this,desc);
}

void RTLib::Ext::GL::GLContext::SetProgram(GLProgram* program){
    if (!program){ return; }
    if (m_Impl->m_Program == program){
        return;
    }
    m_Impl->m_Program = program;
	glUseProgram(m_Impl->m_Program->GetResId());
}

void RTLib::Ext::GL::GLContext::InvalidateProgram()
{
	if (!m_Impl->m_Program) {
		return;
	}
	glUseProgram(0);
	m_Impl->m_Program = nullptr;
}

void RTLib::Ext::GL::GLContext::SetProgramPipeline(GLProgramPipeline* programPipeline)
{
	if (!programPipeline) { return; }
	InvalidateProgram();
	if (m_Impl->m_ProgramPipeline == programPipeline) {
		return;
	}
	glBindProgramPipeline(GLNatives::GetResId(programPipeline));
	m_Impl->m_ProgramPipeline = programPipeline;
}

void RTLib::Ext::GL::GLContext::SetVertexArray(GLVertexArray* vao)
{
    if (!vao){ return;}
    if (m_Impl->m_VertexArray == vao){
        return;
    }
	m_Impl->m_VertexArray = vao;
	glBindVertexArray(m_Impl->m_VertexArray->GetResId());
}

void RTLib::Ext::GL::GLContext::InvalidateVertexArray()
{
	if (!m_Impl->m_VertexArray) {
		return;
	}
	glBindVertexArray(0);
	m_Impl->m_VertexArray = nullptr;
}
