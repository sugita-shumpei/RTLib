#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_RENDER_BUFFER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_RENDER_BUFFER_H
#include "ImplGLBindable.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLRenderbufferBase : public ImplGLBindableBase {
				public:
					friend class ImplGLBindable;
				public:
					virtual ~ImplGLRenderbufferBase()noexcept {}
				protected:
					virtual bool      Create()noexcept override {
						GLuint resId;
						glGenRenderbuffers(1, &resId);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void     Destroy()noexcept {
						GLuint resId = GetResId();
						glGenRenderbuffers(1, &resId);
						SetResId(0);
					}
					virtual void   Bind(GLenum target) {
						GLuint resId = GetResId();
						if (resId > 0) {
							glBindRenderbuffer(target, resId);
						}
					}
					virtual void Unbind(GLenum target) {
						glBindRenderbuffer(target, 0);
					}

				};
				class ImplGLRenderbuffer : public ImplGLBindable {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)->ImplGLRenderbuffer* {
						if (!table || !bPoint) {
							return nullptr;
						}
						auto buffer = new ImplGLRenderbuffer(table, bPoint);
						if (buffer) {
							buffer->InitBase<ImplGLRenderbufferBase>();
							auto res = buffer->Create();
							if (!res) {
								delete buffer;
								return nullptr;
							}
						}
						return buffer;
					}
					virtual ~ImplGLRenderbuffer()noexcept {}
					bool Bind()noexcept {
						return ImplGLBindable::Bind(GL_RENDERBUFFER);
					}
				protected:
					ImplGLRenderbuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept :ImplGLBindable(table, bPoint) {}
				};
			}
		}
	}
}
#endif