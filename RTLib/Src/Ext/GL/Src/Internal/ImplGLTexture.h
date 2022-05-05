#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_TEXTURE_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_TEXTURE_H
#include "ImplGLBindable.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				inline constexpr GLenum  ConvertInternalFormatToBase(GLenum internalFormat) {
					switch (internalFormat) {
					case	GL_R8:	return 	GL_RED;
					case	GL_R8_SNORM:	return 	GL_RED;
					case	GL_R16:	return 	GL_RED;
					case	GL_R16_SNORM:	return 	GL_RED;
					case	GL_RG8:	return 	GL_RG;
					case	GL_RG8_SNORM:	return 	GL_RG;
					case	GL_RG16:	return 	GL_RG;
					case	GL_RG16_SNORM:	return 	GL_RG;
					case	GL_R3_G3_B2:	return 	GL_RGB;
					case	GL_RGB4:	return 	GL_RGB;
					case	GL_RGB5:	return 	GL_RGB;
					case	GL_RGB8:	return 	GL_RGB;
					case	GL_RGB8_SNORM:	return 	GL_RGB;
					case	GL_RGB10:	return 	GL_RGB;
					case	GL_RGB12:	return 	GL_RGB;
					case	GL_RGB16_SNORM:	return 	GL_RGB;
					case	GL_RGBA2:	return 	GL_RGB;
					case	GL_RGBA4:	return 	GL_RGB;
					case	GL_RGB5_A1:	return 	GL_RGBA;
					case	GL_RGBA8:	return 	GL_RGBA;
					case	GL_RGBA8_SNORM:	return 	GL_RGBA;
					case	GL_RGB10_A2:	return 	GL_RGBA;
					case	GL_RGB10_A2UI:	return 	GL_RGBA;
					case	GL_RGBA12:	return 	GL_RGBA;
					case	GL_RGBA16:	return 	GL_RGBA;
					case	GL_SRGB8:	return 	GL_RGB;
					case	GL_SRGB8_ALPHA8:	return 	GL_RGBA;
					case	GL_R16F:	return 	GL_RED;
					case	GL_RG16F:	return 	GL_RG;
					case	GL_RGB16F:	return 	GL_RGB;
					case	GL_RGBA16F:	return 	GL_RGBA;
					case	GL_R32F:	return 	GL_RED;
					case	GL_RG32F:	return 	GL_RG;
					case	GL_RGB32F:	return 	GL_RGB;
					case	GL_RGBA32F:	return 	GL_RGBA;
					case	GL_R11F_G11F_B10F:	return 	GL_RGB;
					case	GL_RGB9_E5:	return 	GL_RGB;
					case	GL_R8I:	return 	GL_RED;
					case	GL_R8UI:	return 	GL_RED;
					case	GL_R16I:	return 	GL_RED;
					case	GL_R16UI:	return 	GL_RED;
					case	GL_R32I:	return 	GL_RED;
					case	GL_R32UI:	return 	GL_RED;
					case	GL_RG8I:	return 	GL_RG;
					case	GL_RG8UI:	return 	GL_RG;
					case	GL_RG16I:	return 	GL_RG;
					case	GL_RG16UI:	return 	GL_RG;
					case	GL_RG32I:	return 	GL_RG;
					case	GL_RG32UI:	return 	GL_RG;
					case	GL_RGB8I:	return 	GL_RGB;
					case	GL_RGB8UI:	return 	GL_RGB;
					case	GL_RGB16I:	return 	GL_RGB;
					case	GL_RGB16UI:	return 	GL_RGB;
					case	GL_RGB32I:	return 	GL_RGB;
					case	GL_RGB32UI:	return 	GL_RGB;
					case	GL_RGBA8I:	return 	GL_RGBA;
					case	GL_RGBA8UI:	return 	GL_RGBA;
					case	GL_RGBA16I:	return 	GL_RGBA;
					case	GL_RGBA16UI:	return 	GL_RGBA;
					case	GL_RGBA32I:	return 	GL_RGBA;
					case	GL_RGBA32UI:	return 	GL_RGBA;
					case	GL_COMPRESSED_RED:	return 	GL_RED;
					case	GL_COMPRESSED_RG:	return 	GL_RG;
					case	GL_COMPRESSED_RGB:	return 	GL_RGB;
					case	GL_COMPRESSED_RGBA:	return 	GL_RGBA;
					case	GL_COMPRESSED_SRGB:	return 	GL_RGB;
					case	GL_COMPRESSED_SRGB_ALPHA:	return 	GL_RGBA;
					case	GL_COMPRESSED_RED_RGTC1:	return 	GL_RED;
					case	GL_COMPRESSED_SIGNED_RED_RGTC1:	return 	GL_RED;
					case	GL_COMPRESSED_RG_RGTC2:	return 	GL_RG;
					case	GL_COMPRESSED_SIGNED_RG_RGTC2:	return 	GL_RG;
					case	GL_COMPRESSED_RGBA_BPTC_UNORM:	return 	GL_RGBA;
					case	GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:	return 	GL_RGBA;
					case	GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:	return 	GL_RGB;
					case	GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:	return 	GL_RGB;
					case    GL_DEPTH_COMPONENT16 : return GL_DEPTH_COMPONENT;
					case    GL_DEPTH_COMPONENT24 : return GL_DEPTH_COMPONENT;
					case    GL_DEPTH_COMPONENT32 : return GL_DEPTH_COMPONENT;
					case    GL_DEPTH_COMPONENT32F: return GL_DEPTH_COMPONENT;
					case    GL_DEPTH24_STENCIL8: return GL_DEPTH_STENCIL;
					case    GL_DEPTH32F_STENCIL8: return GL_DEPTH_STENCIL;
					}
					return GL_RGBA;
				}
				inline constexpr GLenum  QuerySuitableTypeFromFormat(GLenum internalFormat) {
					switch (internalFormat) {
					case	GL_R8:	return 	GL_UNSIGNED_BYTE;
					case	GL_R8_SNORM:	return 	GL_BYTE;
					case	GL_R16:	return 	GL_UNSIGNED_SHORT;
					case	GL_R16_SNORM:	return 	GL_SHORT;
					case	GL_RG8:	return 	GL_UNSIGNED_BYTE;
					case	GL_RG8_SNORM:	return 	GL_BYTE;
					case	GL_RG16:	return 	GL_UNSIGNED_SHORT;
					case	GL_RG16_SNORM:	return 	GL_SHORT;
					case	GL_R3_G3_B2:	return 	GL_UNSIGNED_BYTE_3_3_2;
					case	GL_RGB4:	return  GL_UNSIGNED_BYTE;
					case	GL_RGB5:	return 	GL_UNSIGNED_SHORT;
					case	GL_RGB8:	return 	GL_UNSIGNED_BYTE;
					case	GL_RGB8_SNORM:	return 	GL_UNSIGNED_BYTE;
					case	GL_RGB10:	return 	GL_UNSIGNED_INT;
					case	GL_RGB12:	return 	GL_UNSIGNED_INT;
					case	GL_RGB16_SNORM:	return 	GL_UNSIGNED_SHORT;
					case	GL_RGBA2:	return 	GL_UNSIGNED_BYTE;
					case	GL_RGBA4:	return 	GL_UNSIGNED_SHORT_4_4_4_4;
					case	GL_RGB5_A1:	return 	GL_UNSIGNED_SHORT_5_5_5_1;
					case	GL_RGBA8:	return 	GL_UNSIGNED_BYTE;
					case	GL_RGBA8_SNORM:	return 	GL_BYTE;
					case	GL_RGB10_A2:	return 	GL_UNSIGNED_INT_2_10_10_10_REV;
					case	GL_RGB10_A2UI:	return 	GL_UNSIGNED_INT_2_10_10_10_REV;
					case	GL_RGBA12:	return 	GL_UNSIGNED_BYTE;
					case	GL_RGBA16:	return 	GL_UNSIGNED_SHORT;
					case	GL_SRGB8:	return 	GL_UNSIGNED_BYTE;
					case	GL_SRGB8_ALPHA8:	return 	GL_UNSIGNED_BYTE;
					case	GL_R16F:	return 	GL_HALF_FLOAT;
					case	GL_RG16F:	return 	GL_HALF_FLOAT;
					case	GL_RGB16F:	return 	GL_HALF_FLOAT;
					case	GL_RGBA16F:	return 	GL_HALF_FLOAT;
					case	GL_R32F:	return 	GL_FLOAT;
					case	GL_RG32F:	return 	GL_FLOAT;
					case	GL_RGB32F:	return 	GL_FLOAT;
					case	GL_RGBA32F:	return 	GL_FLOAT;
					case	GL_R11F_G11F_B10F:	return 	GL_UNSIGNED_INT_10F_11F_11F_REV;
					case	GL_RGB9_E5:	return 	GL_UNSIGNED_INT_5_9_9_9_REV;
					case	GL_R8I:	return 	GL_BYTE;
					case	GL_R8UI:	return 	GL_UNSIGNED_BYTE;
					case	GL_R16I:	return 	GL_SHORT;
					case	GL_R16UI:	return 	GL_UNSIGNED_SHORT;
					case	GL_R32I:	return 	GL_INT;
					case	GL_R32UI:	return 	GL_UNSIGNED_INT;
					case	GL_RG8I:	return 	GL_BYTE;
					case	GL_RG8UI:	return 	GL_UNSIGNED_BYTE;
					case	GL_RG16I:	return 	GL_SHORT;
					case	GL_RG16UI:	return 	GL_UNSIGNED_SHORT;
					case	GL_RG32I:	return 	GL_INT;
					case	GL_RG32UI:	return 	GL_UNSIGNED_INT;
					case	GL_RGB8I:	return 	GL_BYTE;
					case	GL_RGB8UI:	return 	GL_UNSIGNED_BYTE;
					case	GL_RGB16I:	return 	GL_SHORT;
					case	GL_RGB16UI:	return 	GL_UNSIGNED_SHORT;
					case	GL_RGB32I:	return 	GL_SHORT;
					case	GL_RGB32UI:	return 	GL_UNSIGNED_INT;
					case	GL_RGBA8I:	return 	GL_BYTE;
					case	GL_RGBA8UI:	return 	GL_UNSIGNED_BYTE;
					case	GL_RGBA16I:	return 	GL_SHORT;
					case	GL_RGBA16UI:return 	GL_UNSIGNED_SHORT;
					case	GL_RGBA32I:	return 	GL_INT;
					case	GL_RGBA32UI:	return 	GL_UNSIGNED_INT;

					case    GL_DEPTH_COMPONENT16: return GL_UNSIGNED_SHORT;
					case    GL_DEPTH_COMPONENT24: return GL_UNSIGNED_INT;
					case    GL_DEPTH_COMPONENT32: return GL_UNSIGNED_INT;
					case    GL_DEPTH_COMPONENT32F: return GL_FLOAT;
					case    GL_DEPTH24_STENCIL8: return GL_UNSIGNED_INT_24_8;
					case    GL_DEPTH32F_STENCIL8: return GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
					}
					return GL_UNSIGNED_BYTE;

				}
				inline constexpr GLsizei CalculatePixelSize(GLenum format, GLenum type) {
					size_t channel = 1;
					size_t count   = 0;
					switch (type) {
					case GL_BYTE:
						channel = 1;
						break;
					case GL_UNSIGNED_BYTE:
						channel = 1;
						break;
					case GL_UNSIGNED_BYTE_3_3_2:
						return (format == GL_RGB) ? 1 : 0;
						break;
					case GL_UNSIGNED_BYTE_2_3_3_REV:
						return (format == GL_RGB) ? 1 : 0;
						break;
					case GL_SHORT:
						channel = 2;
						break;
					case GL_UNSIGNED_SHORT:
						channel = 2;
						break;
					case GL_UNSIGNED_SHORT_5_6_5:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 2 : 0;
						break;
					case GL_UNSIGNED_SHORT_5_6_5_REV:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 2 : 0;
						break;
					case GL_UNSIGNED_SHORT_4_4_4_4:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 2 : 0;
						break;
					case GL_UNSIGNED_SHORT_4_4_4_4_REV:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 2 : 0;
						break;
					case GL_UNSIGNED_SHORT_5_5_5_1:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 2 : 0;
						break;
					case GL_UNSIGNED_SHORT_1_5_5_5_REV:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 2 : 0;
						break;
					case GL_UNSIGNED_INT:
						channel = 4;
						break;
					case GL_UNSIGNED_INT_8_8_8_8:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 4 : 0;
						break;
					case GL_UNSIGNED_INT_8_8_8_8_REV:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 4 : 0;
						break;
					case GL_UNSIGNED_INT_10_10_10_2:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 4 : 0;
						break;
					case GL_UNSIGNED_INT_2_10_10_10_REV:
						return ((format == GL_RGBA) || (format == GL_BGRA)) ? 4 : 0;
						break;
					case GL_INT:
						channel = 4;
						break;
					case GL_HALF_FLOAT:
						channel = 2;
						break;
					case GL_FLOAT:
						channel = 4;
						break;
					}
					switch (format) {
					case GL_RED:
						count = 1;
						break;
					case GL_RG:
						count = 2;
						break;
					case GL_RGB:
						count = 3;
						break;
					case GL_BGR:
						count = 3;
						break;
					case GL_RGBA:
						count = 4;
						break;
					case GL_BGRA:
						count = 4;
						break;
					case GL_RED_INTEGER:
						count = 1;
						break;
					case GL_RG_INTEGER:
						count = 2;
						break;
					case GL_RGB_INTEGER:
						count = 3;
						break;
					case GL_BGR_INTEGER:
						count = 3;
						break;
					case GL_RGBA_INTEGER:
						count = 4;
						break;
					case GL_BGRA_INTEGER:
						count = 4;
						break;
					case GL_STENCIL_INDEX:
						count = 1;
						break;
					case GL_DEPTH_COMPONENT:
						count = 1;
						break;
					case GL_DEPTH_STENCIL:
						count = 1;
						break;
					default:
						count = 1;
						break;
					}
					return channel * count;
				}
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

					bool CopyFromMemory(const void* pData, GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint  dstXOffset = 0, GLint  dstYOffset = 0, GLint  dstZOffset = 0);
					bool CopyToMemory(        void* pData, GLenum format, GLenum type, GLint level);

					bool CopyFromBuffer(ImplGLBuffer* src  , GLenum format, GLenum type, GLint level, GLint layer = 0, GLsizei layers = 1, GLsizei width = 1, GLsizei height = 1, GLsizei depth = 1, GLint  dstXOffset = 0, GLint  dstYOffset = 0, GLint  dstZOffset = 0, GLintptr srcOffset = 0);
					bool CopyToBuffer(  ImplGLBuffer* src  , GLenum format, GLenum type, GLint level, GLintptr srcOffset = 0);
				protected:
					ImplGLTexture(GLenum target, ImplGLResourceTable* table, ImplGLBindingPoint* tPoint, ImplGLBindingPoint* bPoint)noexcept :ImplGLBindable(table, tPoint), m_Target{ target }, m_BPBuffer{bPoint}{}
					auto GetTxTarget()const noexcept -> GLuint { return m_Target; }
					auto GetBPBuffer()const noexcept -> const ImplGLBindingPoint*;
					auto GetMipWidth (GLint level)const noexcept -> GLsizei;
					auto GetMipHeight(GLint level)const noexcept -> GLsizei;
					auto GetMipDepth (GLint level)const noexcept -> GLsizei;
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