#ifndef RTLIB_EXT_CUGL_CUGL_BUFFER_H
#define RTLIB_EXT_CUGL_CUGL_BUFFER_H
#include <RTLib/Ext/CUGL/CUGLContext.h>
#include <RTLib/Ext/CUGL/CUGLCommon.h>
#include <RTLib/Ext/CUGL/UuidDefinitions.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAStream.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLContext.h>
namespace RTLib
{
	namespace Ext
	{
		namespace CUGL
		{
			class CUGLBuffer : public Core::Buffer
			{
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(CUGLBuffer, Core::Buffer, RTLIB_TYPE_UUID_RTLIB_EXT_CUGL_CUGL_BUFFER);
				auto New(CUGLContext* ctx, GL::GLBuffer* bufferGL, unsigned int flags = CUGLGraphicsRegisterFlagsNone)->CUGLBuffer*;
				virtual  ~CUGLBuffer()noexcept;
				virtual void Destroy()noexcept;
				
				void   MapCUDA(CUDA::CUDAStream* stream = nullptr)
				{
					if (m_Impl->isMapped) { return; }
					cuGraphicsMapResources(1,&m_Impl->graphicsResource, )
				}
				void UnmapCUDA()
				{
				}
			private:
				CUGLBuffer(CUGLContext* ctx, GL::GLBuffer* bufferGL)noexcept;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
				
			};
		}
	}
}
#endif
