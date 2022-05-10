#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_RENDER_BUFFER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_RENDER_BUFFER_H
#include "ImplGLBindable.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLRenderbuffer : public ImplGLBindable {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)->ImplGLRenderbuffer*;
					virtual ~ImplGLRenderbuffer()noexcept;
					bool Bind()noexcept;
				protected:
					ImplGLRenderbuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept;
				};
			}
		}
	}
}
#endif