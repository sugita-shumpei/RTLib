#ifndef RTLIB_EXT_CUDA_CUDA_CONTEXT_H
#define RTLIB_EXT_CUDA_CUDA_CONTEXT_H
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <RTLib/Core/Context.h>
#include <vector>
namespace RTLib {
	namespace Ext
	{
		namespace CUDA
		{
			class CUDABuffer ;
			class CUDAImage  ;
			class CUDATexture;
			class CUDAModule;
			class CUDAStream;
			class CUDAContext : public RTLib::Core::Context
			{
			public:
				virtual ~CUDAContext()noexcept;
				// Context ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
				virtual bool Initialize() override;
				virtual void Terminate() override;

				bool MakeContextCurrent();

				auto CreateBuffer (const CUDABufferDesc      & desc)->CUDABuffer *;
				auto CreateImage  (const CUDAImageDesc       & desc)->CUDAImage  *;
				auto CreateTexture(const CUDATextureImageDesc& desc)->CUDATexture*;
				auto CreateStream ()->CUDAStream*;

				auto LoadModuleFromFile(const char* filename)->CUDAModule*;
				auto LoadModuleFromData(const void* data, const std::vector<CUDAJitOptionValue>& optionValues = {})->CUDAModule*;
				/*Copy*/
				bool CopyBuffer(CUDABuffer* srcBuffer, CUDABuffer* dstBuffer, const std::vector<CUDABufferCopy>& regions);
				bool CopyMemoryToBuffer(CUDABuffer* buffer, const std::vector<CUDAMemoryBufferCopy>& regions);
				bool CopyBufferToMemory(CUDABuffer* buffer, const std::vector<CUDABufferMemoryCopy>& regions);
				bool CopyImageToBuffer(CUDAImage * srcImage, CUDABuffer* dstBuffer, const std::vector<CUDABufferImageCopy>& regions);
				bool CopyBufferToImage(CUDABuffer* srcBuffer,CUDAImage* dstImage, const std::vector<CUDABufferImageCopy>& regions);
				bool CopyImageToMemory(CUDAImage* image, const std::vector<CUDAImageMemoryCopy>& regions);
				bool CopyMemoryToImage(CUDAImage* image, const std::vector<CUDAMemoryImageCopy>& regions);
			private:
				CUcontext m_CtxCU = nullptr;
				CUdevice  m_DevCU = 0;
			};
		}
	}
}
#endif
