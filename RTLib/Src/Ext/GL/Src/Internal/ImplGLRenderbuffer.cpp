#include "ImplGLRenderbuffer.h"
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
#ifndef NDEBUG
						glBindRenderbuffer(target, 0);
#endif
					}

				};
				auto ImplGLRenderbuffer::New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) -> ImplGLRenderbuffer* {
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
				ImplGLRenderbuffer::~ImplGLRenderbuffer() noexcept {}
				bool ImplGLRenderbuffer::Bind() noexcept {
					return ImplGLBindable::Bind(GL_RENDERBUFFER);
				}
				ImplGLRenderbuffer::ImplGLRenderbuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) noexcept :ImplGLBindable(table, bPoint) {}
			}
		}
	}
}