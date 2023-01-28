#ifndef RTLIB_EXT_CUGL_CUGL_IMAGE_H
#define RTLIB_EXT_CUGL_CUGL_IMAGE_H
#include <RTLib/Ext/CUGL/CUGLCommon.h>
#include <RTLib/Ext/CUGL/UuidDefinitions.h>
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <RTLib/Ext/CUDA/CUDAStream.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLNatives.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <memory>
namespace RTLib
{
	namespace Ext
	{
		namespace CUGL
		{
			class CUDA::CUDAContext;
			class CUGLImage : public Core::Image
			{
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(CUGLImage, Core::Image, RTLIB_TYPE_UUID_RTLIB_EXT_CUGL_CUGL_IMAGE);
				static auto New(CUDA::CUDAContext* ctx, GL::GLImage* imageGL, unsigned int flags = CUGLGraphicsRegisterFlagsNone)->CUGLImage*;
				virtual  ~CUGLImage()noexcept;
				virtual void Destroy()noexcept override;

				auto MapSubImage(uint32_t layer, uint32_t level, CUDA::CUDAStream* stream = nullptr)->CUDA::CUDAImage*;
				auto MapMipImage(CUDA::CUDAStream* stream = nullptr)->CUDA::CUDAImage*;
				void Unmap(CUDA::CUDAStream* stream = nullptr);

				auto GetContextCU()noexcept -> CUDA::CUDAContext*;
				auto GetContextCU()const noexcept -> const CUDA::CUDAContext*;
			private:
				CUGLImage(CUDA::CUDAContext* ctx, GL::GLImage* imageGL)noexcept;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;

			};
		}
	}
}
#endif
