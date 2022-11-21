#ifndef RTLIB_BACKENDS_OGL4_OGL4_CONTEXT_H
#define RTLIB_BACKENDS_OGL4_OGL4_CONTEXT_H
#include <vector>
#include <memory>
namespace RTLib
{
	namespace Window {
		class Window;
	}
	namespace Backends
	{
		namespace Ogl4 {
			enum ClearBufferMask {
				ClearBufferUnknown = 0,
				ClearColorBuffer   = 1<<0,
				ClearDepthBuffer   = 1<<1,
				ClearStencilBuffer = 1<<2,
			};
			enum BufferTargetMask:unsigned int
			{
				BufferTargetUnknown=0,
				BufferTargetVertex=1<<0,
				BufferTargetAtomicCounter=1<<1,
				BufferTargetCopySrc=1<<2,
				BufferTargetCopyDst=1<<3,
				BufferTargetDispatchIndirect=1<<4,
				BufferTargetDrawIndirect=1<<5,
				BufferTargetIndex=1<<6,
				BufferTargetPixelCopySrc=1<<7,
				BufferTargetPixelCopyDst=1<<8,
				BufferTargetQuery=1<<9,
				BufferTargetShaderStorage=1<<10,
				BufferTargetTexture=1<<11,
				BufferTargetTransformFeedback=1<<12,
				BufferTargetUniform=1<<13,
			};
			enum ShaderStageMask
			{
				ShaderStageUnknown = 0,
				ShaderStageCompute = 1<<0,
				ShaderStageVertex  = 1<<1,
				ShaderStageTessControl = 1<<2,
				ShaderStageTessEvaluation = 1<<3,
				ShaderStageGeometry = 1<<4,
				ShaderStageFragment = 1<<5,
			};
			enum class DepthCompareOp
			{
				eNever = 0,
				eLess,
				eLessEqual,
				eEqual,
				eGreater,
				eNotEqual,
				eGreaterEqual,
				eAlways
			};

			using PfnGlProc         = void(*)(void);
			using PfnGetProcAddress = auto(*)(const char*)->PfnGlProc;
			class CurrentContext;
			class Context {
			public:
				 Context(RTLib::Window::Window* window, PfnGetProcAddress getProcAddress) noexcept;
				~Context() noexcept;

				 Context(const Context&) = delete;
				 Context(Context&&)      = delete;
				 Context& operator=(const Context&) = delete;
				 Context& operator=(     Context&&) = delete;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			class CurrentContext {

			};
		}
	}
}
#endif
