#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_TEXTURE_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_TEXTURE_H
#include "ImplGLBindable.h"
#include <array>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {

				class ImplGLBuffer;
				class ImplGLTextureBase : public ImplGLBindableBase {
				public:
					friend class ImplGLBindable;
				public:
					virtual ~ImplGLTextureBase()noexcept {}
				protected:
					virtual bool      Create()noexcept override {
						GLuint resId;
						glGenTextures(1, &resId);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void     Destroy()noexcept {
						GLuint resId = GetResId();
						glDeleteTextures(1, &resId);
						SetResId(0);
					}
					virtual void   Bind(GLenum target) {
						GLuint resId = GetResId();
						if (resId > 0) {
							glBindTexture(target, resId);
						}
					}
					virtual void Unbind(GLenum target) {
						glBindTexture(target, 0);
					}

				};
				class ImplGLTexture : public ImplGLBindable {
				public:
					friend class ImplGLBuffer;
					friend class ImplGLFramebuffer;
				public:
					static auto New(GLenum target, ImplGLResourceTable* table, ImplGLBindingPoint* tPoint, ImplGLBindingPoint* bPoint)->ImplGLTexture* {
						if (!table || !bPoint) {
							return nullptr;
						}
						auto buffer = new ImplGLTexture(target,table, tPoint, bPoint);
						if (buffer) {
							buffer->InitBase<ImplGLTextureBase>();
							auto res = buffer->Create();
							if (!res) {
								delete buffer;
								return nullptr;
							}
						}
						return buffer;
					}
					virtual ~ImplGLTexture()noexcept {}
					bool Bind()noexcept {
						return ImplGLBindable::Bind(m_Target);
					}

					bool Allocate(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height, GLsizei depth);

					bool IsAllocated()const noexcept { return m_AllocationInfo != std::nullopt; }

					bool CopyImageFromMemory(const void* pData, GLenum format  , GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint  dstXOffset = 0, GLint  dstYOffset = 0, GLint  dstZOffset = 0);
					bool CopyImageToMemory  (      void* pData, GLenum format  , GLenum type, GLint level);

					bool CopyImageFromBuffer(ImplGLBuffer* src  , GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint  dstXOffset = 0, GLint  dstYOffset = 0, GLint  dstZOffset = 0, GLintptr srcOffset = 0);
					bool CopyImageToBuffer(  ImplGLBuffer* src  , GLenum format, GLenum type, GLint level, GLintptr srcOffset = 0);
					
					bool CopyFaceImageFromMemory(GLenum target, const void* pData, GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint  dstXOffset = 0, GLint  dstYOffset = 0, GLint  dstZOffset = 0);
					bool CopyFaceImageToMemory(  GLenum target, void* pData, GLenum format, GLenum type, GLint level);

					bool CopyFaceImageFromBuffer(GLenum target, ImplGLBuffer* src, GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint  dstXOffset = 0, GLint  dstYOffset = 0, GLint  dstZOffset = 0, GLintptr srcOffset = 0);
					bool CopyFaceImageToBuffer(  GLenum target, ImplGLBuffer* src, GLenum format, GLenum type, GLint level, GLintptr srcOffset = 0);
				protected:
					ImplGLTexture(GLenum target, ImplGLResourceTable* table, ImplGLBindingPoint* tPoint, ImplGLBindingPoint* bPoint)noexcept :ImplGLBindable(table, tPoint), m_Target{ target }, m_BPBuffer{bPoint}{}
					auto GetTxTarget()const noexcept -> GLuint { return m_Target; }
					auto GetBPBuffer()const noexcept -> const ImplGLBindingPoint*;
					auto GetMipWidth (GLint level)const noexcept -> GLsizei;
					auto GetMipHeight(GLint level)const noexcept -> GLsizei;
					auto GetMipDepth (GLint level)const noexcept -> GLsizei;
				private:
					//ALLOCATE
					void AllocateTexture1D(GLenum internalFormat, GLint levels, GLsizei width);
					void AllocateTexture2D(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height);
					void AllocateTexture3D(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height, GLsizei depth);
					void AllocateTexture1DArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width);
					void AllocateTexture2DArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height);
					void AllocateTextureCubemap(GLenum internalFormat, GLint levels, GLsizei width, GLsizei height);
					void AllocateTextureCubemapArray(GLenum internalFormat, GLint levels, GLsizei layers, GLsizei width, GLsizei height);
					//COPYIMAGE
					bool CopyImage1DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLsizei width, GLint  dstXOffset);
					bool CopyImage2DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLsizei width, GLsizei height, GLint  dstXOffset, GLint  dstYOffset);
					bool CopyImage3DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLsizei width, GLsizei height, GLsizei depth, GLint  dstXOffset, GLint  dstYOffset, GLint  dstZOffset);
					bool CopyLayeredImage1DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLint  dstXOffset);
					bool CopyLayeredImage2DFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLint layer, GLsizei layers, GLsizei width, GLsizei height, GLint  dstXOffset, GLint  dstYOffset);
				private:
					struct AllocationInfo {
						GLint   levels;
						GLenum  internalFormat;
						GLsizei layers;
						GLsizei width;
						GLsizei height;
						GLsizei depth;
					};
					GLenum m_Target;
					std::optional<AllocationInfo> m_AllocationInfo = std::nullopt;
					ImplGLBindingPoint* m_BPBuffer;
				};
			}
		}
	}
}
#endif