#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_SAMPLER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_SAMPLER_H
#include "ImplGLBindable.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLSampler : public ImplGLBindable {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)->ImplGLSampler*;
					virtual ~ImplGLSampler()noexcept;
					bool   Bind(GLuint unit);
				protected:
					ImplGLSampler(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept;
				};
			}
		}
	}
}
#endif