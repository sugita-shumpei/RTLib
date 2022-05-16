#ifndef RTLIB_EXT_GL_GL_BUFFER_H
#define RTLIB_EXT_GL_GL_BUFFER_H
#include <RTLib/Ext/GL/GLCommon.h>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			class GLContext;
			class GLBuffer {
			public:
				static auto Allocate(GLContext* context,const GLBufferDesc& desc)->GLBuffer*;
				virtual ~GLBuffer()noexcept{}

				void Destroy();
				auto GetBufferUsage   ()const noexcept -> GLBufferUsageFlags          { return m_Usage;    }
			private:
				GLBuffer(GLuint resId, GLBufferUsageFlags usage)noexcept;
				auto GetResId()const noexcept -> GLuint { return m_ResId; }
			private:
				GLContext*                  m_Context ;
				GLuint                      m_ResId   ;
				GLBufferUsageFlags          m_Usage   ;
			};
		}
	}
}
#endif
