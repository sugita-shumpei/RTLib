#ifndef RTLIB_EXT_CUGL_CUGL_BUFFER_H
#define RTLIB_EXT_CUGL_CUGL_BUFFER_H
#include <RTLib/Ext/CUGL/CUGLCommon.h>
#include <RTLib/Ext/CUGL/UuidDefinitions.h>
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAStream.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLNatives.h>
#include <RTLib/Ext/GL/GLContext.h>
namespace RTLib
{
	namespace Ext
	{
		namespace CUGL
		{
			class CUDA::CUDAContext;
			class CUGLBuffer : public Core::Buffer
			{
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(CUGLBuffer, Core::Buffer, RTLIB_TYPE_UUID_RTLIB_EXT_CUGL_CUGL_BUFFER);
				static auto New(CUDA::CUDAContext* ctx, GL::GLBuffer* bufferGL, unsigned int flags = CUGLGraphicsRegisterFlagsNone)->CUGLBuffer*;
				virtual  ~CUGLBuffer()noexcept;
				virtual void Destroy()noexcept override;
				
				auto   Map(CUDA::CUDAStream* stream = nullptr)->CUDA::CUDABuffer*;
				void Unmap(CUDA::CUDAStream* stream = nullptr);

				auto GetContextCU()noexcept -> CUDA::CUDAContext*;
				auto GetContextCU()const noexcept -> const CUDA::CUDAContext*;
			private:
				CUGLBuffer(CUDA::CUDAContext* ctx, GL::GLBuffer* bufferGL)noexcept;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
				
			};
		}
	}
}
#endif
