#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_FRAME_BUFFER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_FRAME_BUFFER_H
#include "ImplGLBindable.h"
#include "ImplGLTexture.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLFramebufferBase : public ImplGLBindableBase {
				public:
					friend class ImplGLBindable;
				public:
					virtual ~ImplGLFramebufferBase()noexcept {}
				protected:
					virtual bool      Create()noexcept override {
						GLuint resId;
						glGenFramebuffers(1, &resId);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void     Destroy()noexcept {
						GLuint resId = GetResId();
						glDeleteFramebuffers(1, &resId);
						SetResId(0);
					}
					virtual void   Bind(GLenum target) {
						GLuint resId = GetResId();
						if (resId > 0) {
							glBindFramebuffer(target, resId);
						}
					}
					virtual void Unbind(GLenum target) {
						glBindFramebuffer(target, 0);
					}

				};
				class ImplGLFramebuffer : public ImplGLBindable {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)->ImplGLFramebuffer* {
						if (!table || !bPoint) {
							return nullptr;
						}
						auto buffer = new ImplGLFramebuffer(table, bPoint);
						if (buffer) {
							buffer->InitBase<ImplGLFramebufferBase>();
							auto res = buffer->Create();
							if (!res) {
								delete buffer;
								return nullptr;
							}
						}
						return buffer;
					}
					virtual ~ImplGLFramebuffer()noexcept {}
					bool Bind(GLenum target)noexcept {
						return ImplGLBindable::Bind(target);
					}

					bool AttachColorTexture(GLuint  idx, ImplGLTexture* texture, GLint level = 0, GLint layer = 0);
					bool AttachDepthTexture(ImplGLTexture* texture, GLint level = 0, GLint layer = 0);
					bool AttachStencilTexture(ImplGLTexture* texture, GLint level = 0, GLint layer = 0);
					bool AttachDepthStencilTexture(ImplGLTexture* texture, GLint level = 0, GLint layer = 0);

					bool AttachColorRenderbuffer(GLuint idx, GLint level);
					bool AttachDepthRenderbuffer(GLint level);
					bool AttachStencilRenderbuffer(GLint level);
					bool AttachDepthStencilRenderbuffer(GLint level);

					bool IsCompleted()const noexcept;
				protected:
					ImplGLFramebuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept :ImplGLBindable(table, bPoint) {}
				private:
					std::vector<ImplGLBindable*> m_Bindables;
				};
			}
		}
	}
}
#endif