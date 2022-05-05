#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_CONTEXT_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_CONTEXT_H
#include "ImplGLBindable.h"
#include "ImplGLResource.h"
#include "ImplGLBuffer.h"
#include "ImplGLTexture.h"
#include "ImplGLSampler.h"
#include "ImplGLFramebuffer.h"
#include "ImplGLRenderbuffer.h"
#include <glad/glad.h>
#include <unordered_map>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLContext {
				public:
					static auto New() ->ImplGLContext* {
						auto ptr = new ImplGLContext();
						ptr->Init();
						return ptr;
					}
					virtual ~ImplGLContext()noexcept {}

					auto CreateBuffer ()             ->ImplGLBuffer*;
					auto CreateTexture(GLenum target)->ImplGLTexture*;
					auto CreateSampler()             ->ImplGLSampler*;
					auto CreateFramebuffer()         ->ImplGLFramebuffer*;
					auto CreateRenderbuffer()        ->ImplGLRenderbuffer*;
				private:
					ImplGLContext()noexcept {}
					void Init() {
						m_BPBuffer.AddTarget(GL_ARRAY_BUFFER);
						m_BPBuffer.AddTarget(GL_ATOMIC_COUNTER_BUFFER);
						m_BPBuffer.AddTarget(GL_COPY_READ_BUFFER);
						m_BPBuffer.AddTarget(GL_COPY_WRITE_BUFFER);
						m_BPBuffer.AddTarget(GL_DISPATCH_INDIRECT_BUFFER);
						m_BPBuffer.AddTarget(GL_DRAW_INDIRECT_BUFFER);
						m_BPBuffer.AddTarget(GL_ELEMENT_ARRAY_BUFFER);
						m_BPBuffer.AddTarget(GL_PIXEL_PACK_BUFFER);
						m_BPBuffer.AddTarget(GL_PIXEL_UNPACK_BUFFER);
						m_BPBuffer.AddTarget(GL_QUERY_BUFFER);
						m_BPBuffer.AddTarget(GL_SHADER_STORAGE_BUFFER);
						m_BPBuffer.AddTarget(GL_TEXTURE_BUFFER);
						m_BPBuffer.AddTarget(GL_TRANSFORM_FEEDBACK_BUFFER);
						m_BPBuffer.AddTarget(GL_UNIFORM_BUFFER);

						m_BPTexture.AddTarget(GL_TEXTURE_1D);
						m_BPTexture.AddTarget(GL_TEXTURE_2D);
						m_BPTexture.AddTarget(GL_TEXTURE_3D);
						m_BPTexture.AddTarget(GL_TEXTURE_1D_ARRAY);
						m_BPTexture.AddTarget(GL_TEXTURE_2D_ARRAY);
						m_BPTexture.AddTarget(GL_TEXTURE_2D_MULTISAMPLE);
						m_BPTexture.AddTarget(GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
						m_BPTexture.AddTarget(GL_TEXTURE_RECTANGLE);
						m_BPTexture.AddTarget(GL_TEXTURE_CUBE_MAP);
						m_BPTexture.AddTarget(GL_TEXTURE_CUBE_MAP_ARRAY);
						m_BPTexture.AddTarget(GL_TEXTURE_BUFFER);

						m_BPFramebuffer.AddTarget(GL_FRAMEBUFFER);
						m_BPFramebuffer.AddTarget(GL_DRAW_FRAMEBUFFER);
						m_BPFramebuffer.AddTarget(GL_READ_FRAMEBUFFER);

						m_BPRenderbuffer.AddTarget(GL_RENDERBUFFER);

						GLint maxTexUnits;
						glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxTexUnits);
						for (uint32_t i = 0; i < maxTexUnits; ++i) {
							m_BPSampler.AddTarget(i);
						}
					}
				private:
					ImplGLResourceTable m_ResourceTable     = {};
					ImplGLBindingPoint  m_BPBuffer          = {};
					ImplGLBindingPoint  m_BPTexture         = {};
					ImplGLBindingPoint  m_BPSampler         = {};
					ImplGLBindingPoint  m_BPFramebuffer     = {};
					ImplGLBindingPoint  m_BPRenderbuffer    = {};
				};
			}
		}
	}
}
#endif