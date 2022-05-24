#ifndef RTLIB_EXT_CUGL_CUGL_CONTEXT_H
#define RTLIB_EXT_CUGL_CUGL_CONTEXT_H
#include <RTLib/Ext/CUGL/UuidDefinitions.h>
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/GL/GLContext.h>
namespace RTLib
{
	namespace Ext
	{
		namespace CUGL
		{
			class CUGLContext : public Core::Context
			{
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(CUGLContext, Core::Context, RTLIB_TYPE_UUID_RTLIB_EXT_CUGL_CUGL_CONTEXT);
				virtual ~CUGLContext()noexcept;

				virtual bool Initialize() override;
				virtual void Terminate() override;

				auto GetContextCUDA()const noexcept -> const CUDA::CUDAContext* { return m_CtxCUDA; }
				auto GetContextCUDA()      noexcept ->       CUDA::CUDAContext* { return m_CtxCUDA; }
				auto GetContextGL()  const noexcept -> const     GL::GLContext* { return m_CtxGL;   }
				auto GetContextGL()        noexcept ->           GL::GLContext* { return m_CtxGL;   }
			private:
				CUGLContext(CUDA::CUDAContext* ctxCUDA, GL::GLContext* ctxGL)noexcept:m_CtxCUDA{ctxCUDA}, m_CtxGL{ctxGL}{}
			private:
				CUDA::CUDAContext* m_CtxCUDA;
				GL::GLContext*     m_CtxGL;
			};
		}
	}
}
#endif
