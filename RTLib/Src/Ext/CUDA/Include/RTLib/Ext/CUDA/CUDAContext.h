#ifndef RTLIB_EXT_CUDA_CUDA_CONTEXT_H
#define RTLIB_EXT_CUDA_CUDA_CONTEXT_H
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <RTLib/Core/Context.h>
namespace RTLib {
	namespace Ext
	{
		namespace CUDA
		{
			class CUDABuffer;
			class CUDAImage;
			class CUDAContext : public RTLib::Core::Context
			{
			public:
				virtual ~CUDAContext()noexcept;
				// Context ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
				virtual bool Initialize() override;
				virtual void Terminate() override;

				auto CreateBuffer(const CUDABufferDesc& desc)->CUDABuffer*;
				auto CreateImage (const CUDAImageDesc&  desc)->CUDAImage *;
			private:
				CUcontext m_CtxCU = nullptr;
				CUdevice  m_DevCU = 0;
			};
		}
	}
}
#endif
