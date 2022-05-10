#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_FRAME_BUFFER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_FRAME_BUFFER_H
#include "ImplGLBindable.h"
#include "ImplGLTexture.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLFramebuffer : public ImplGLBindable {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)->ImplGLFramebuffer*;
					virtual ~ImplGLFramebuffer()noexcept;
					bool Bind(GLenum target);

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
					ImplGLFramebuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept;
				private:
					std::vector<ImplGLBindable*> m_Bindables;
				};
			}
		}
	}
}
#endif