#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_TEXTURE_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_TEXTURE_H
#include "ImplGLBindable.h"
#include "ImplGLUtils.h"
#include <iostream>
#include <array>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			namespace Internal
			{
				class ImplGLBuffer;
				class ImplGLTexture : public ImplGLBindable
				{
				public:
					friend class ImplGLBuffer;
					friend class ImplGLFramebuffer;

				public:
					static auto New(GLenum target, ImplGLResourceTable* table, ImplGLBindingPoint* tPoint, const ImplGLBindingPoint* bPoint)->ImplGLTexture*;
					virtual ~ImplGLTexture() noexcept;
					bool Bind();
					void Unbind() noexcept;

					bool Allocate(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth);

					bool IsAllocated() const noexcept;

					bool CopyImageFromMemory(const void *pData, GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint dstXOffset = 0, GLint dstYOffset = 0, GLint dstZOffset = 0);
					bool CopyImageToMemory(void *pData, GLenum format, GLenum type, GLint level);

					bool CopyImageFromBuffer(ImplGLBuffer *src, GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint dstXOffset = 0, GLint dstYOffset = 0, GLint dstZOffset = 0, GLintptr srcOffset = 0);
					bool CopyImageToBuffer(ImplGLBuffer *src, GLenum format, GLenum type, GLint level, GLintptr srcOffset = 0);

					bool CopyFaceImageFromMemory(GLenum target, const void *pData, GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint dstXOffset = 0, GLint dstYOffset = 0, GLint dstZOffset = 0);
					bool CopyFaceImageToMemory(GLenum target, void *pData, GLenum format, GLenum type, GLint level);

					bool CopyFaceImageFromBuffer(GLenum target, ImplGLBuffer *src, GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint dstXOffset = 0, GLint dstYOffset = 0, GLint dstZOffset = 0, GLintptr srcOffset = 0);
					bool CopyFaceImageToBuffer(GLenum target, ImplGLBuffer *src, GLenum format, GLenum type, GLint level, GLintptr srcOffset = 0);

				protected:
					ImplGLTexture(GLenum target, ImplGLResourceTable* table, ImplGLBindingPoint* tPoint, const ImplGLBindingPoint* bPoint) noexcept;
					auto GetTxTarget() const noexcept -> GLuint;
					auto GetBPBuffer() const noexcept -> const ImplGLBindingPoint *;
					auto GetMipWidth(GLint level) const noexcept -> GLsizei;
					auto GetMipHeight(GLint level) const noexcept -> GLsizei;
					auto GetMipDepth(GLint level) const noexcept -> GLsizei;

				private:
					// ALLOCATE
					bool AllocateTexture1D(GLenum internalFormat, GLint levels, GLsizei width);
					bool AllocateTexture2D(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height);
					bool AllocateTexture3D(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height, GLsizei depth);
					bool AllocateTexture1DArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width);
					bool AllocateTexture2DArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height);
					bool AllocateTextureCubemap(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height);
					bool AllocateTextureCubemapArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height);
					// COPYIMAGE
					bool CopyImage1DFromMemory(const void *pData, GLenum format, GLenum type, GLint level, GLsizei width, GLint dstXOffset);
					bool CopyImage2DFromMemory(const void *pData, GLenum format, GLenum type, GLint level, GLsizei width, GLsizei height, GLint dstXOffset, GLint dstYOffset);
					bool CopyImage3DFromMemory(const void *pData, GLenum format, GLenum type, GLint level, GLsizei width, GLsizei height, GLsizei depth, GLint dstXOffset, GLint dstYOffset, GLint dstZOffset);
					bool CopyLayeredImage1DFromMemory(const void *pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLint dstXOffset);
					bool CopyLayeredImage2DFromMemory(const void *pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLint dstXOffset, GLint dstYOffset);
					bool CopyFaceImage2DFromMemory(GLenum target, const void *pData, GLenum format, GLenum type, GLint level, GLsizei width, GLsizei height, GLint dstXOffset, GLint dstYOffset);
					bool CopyLayeredFaceImage2DFromMemory(GLenum target, const void *pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLint dstXOffset, GLint dstYOffset);
					static inline constexpr bool IsCubeFaceTarget(GLenum target)
					{
						constexpr GLenum cubeFaceTargets[] = {
							GL_TEXTURE_CUBE_MAP_POSITIVE_X,
							GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
							GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
							GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
							GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
							GL_TEXTURE_CUBE_MAP_NEGATIVE_Z};
						for (auto &cubeFaceTarget : cubeFaceTargets)
						{
							if (target == cubeFaceTarget)
							{
								return true;
							}
						}
						return false;
					}

				private:
					struct AllocationInfo
					{
						GLint levels;
						GLenum internalFormat;
						GLsizei layers;
						GLsizei width;
						GLsizei height;
						GLsizei depth;
					};
					GLenum m_Target;
					std::optional<AllocationInfo> m_AllocationInfo = std::nullopt;
					const ImplGLBindingPoint *m_BPBuffer;
				};
			}
		}
	}
}
#endif