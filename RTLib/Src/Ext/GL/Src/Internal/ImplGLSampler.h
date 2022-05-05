#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_SAMPLER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_SAMPLER_H
#include "ImplGLBindable.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLSamplerBase : public ImplGLBindableBase {
				public:
					friend class ImplGLBindable;
				public:
					virtual ~ImplGLSamplerBase()noexcept {}
				protected:
					virtual bool      Create()noexcept override {
						GLuint resId;
						glGenSamplers(1, &resId);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void     Destroy()noexcept {
						GLuint resId = GetResId();
						glDeleteSamplers(1, &resId);
						SetResId(0);
					}
					virtual void   Bind(GLenum target) {
						GLuint resId = GetResId();
						if (resId > 0) {
							glBindSampler(target, resId);
						}
					}
					virtual void Unbind(GLenum target) {
						glBindSampler(target, 0);
					}

				};
				class ImplGLSampler : public ImplGLBindable {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)->ImplGLSampler* {
						if (!table || !bPoint) {
							return nullptr;
						}
						auto buffer = new ImplGLSampler(table, bPoint);
						if (buffer) {
							buffer->InitBase<ImplGLSamplerBase>();
							auto res = buffer->Create();
							if (!res) {
								delete buffer;
								return nullptr;
							}
						}
						return buffer;
					}
					virtual ~ImplGLSampler()noexcept {}
					bool   Bind(GLuint unit) {
						return ImplGLBindable::Bind(unit);
					}
				protected:
					ImplGLSampler(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept :ImplGLBindable(table, bPoint) {}
				};
			}
		}
	}
}
#endif