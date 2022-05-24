#ifndef RTLIB_EXT_CUDA_CUDA_CONTEXT_H
#define RTLIB_EXT_CUDA_CUDA_CONTEXT_H
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <RTLib/Ext/CUDA/UuidDefinitions.h>
#include <RTLib/Core/Context.h>
#include <vector>
namespace RTLib
{
	namespace Ext
	{
		namespace CUDA
		{
			class CUDABuffer;
			class CUDAImage;
			class CUDATexture;
			class CUDAModule;
			class CUDAStream;
			class CUDANatives;
			class CUDAContext : public Core::Context
			{
				friend class CUDANatives;
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(CUDAContext, Core::Context, RTLIB_TYPE_UUID_RTLIB_EXT_CUDA_CUDA_CONTEXT);
				virtual ~CUDAContext()noexcept;
				// Context を介して継承されました
				virtual bool Initialize() override;
				virtual void Terminate() override;

				bool MakeContextCurrent();

				auto CreateBuffer(const CUDABufferCreateDesc &desc) -> CUDABuffer *;
				auto CreateImage(const CUDAImageCreateDesc &desc) -> CUDAImage *;
				auto CreateTexture(const CUDATextureImageCreateDesc &desc) -> CUDATexture *;
				auto CreateStream() -> CUDAStream *;

				auto LoadModuleFromFile(const char *filename) -> CUDAModule *;
				auto LoadModuleFromData(const void *data, const std::vector<CUDAJitOptionValue> &optionValues = {}) -> CUDAModule *;
				/*Copy*/
				bool CopyBuffer(CUDABuffer *srcBuffer, CUDABuffer *dstBuffer, const std::vector<CUDABufferCopy> &regions);
				bool CopyMemoryToBuffer(CUDABuffer *buffer, const std::vector<CUDAMemoryBufferCopy> &regions);
				bool CopyBufferToMemory(CUDABuffer *buffer, const std::vector<CUDABufferMemoryCopy> &regions);
				bool CopyImageToBuffer(CUDAImage *srcImage, CUDABuffer *dstBuffer, const std::vector<CUDABufferImageCopy> &regions);
				bool CopyBufferToImage(CUDABuffer *srcBuffer, CUDAImage *dstImage, const std::vector<CUDABufferImageCopy> &regions);
				bool CopyImageToMemory(CUDAImage *image, const std::vector<CUDAImageMemoryCopy> &regions);
				bool CopyMemoryToImage(CUDAImage *image, const std::vector<CUDAMemoryImageCopy> &regions);
			protected:
				auto GetCUcontext() const noexcept -> CUcontext { return m_CtxCU; }
				auto GetCUdevice () const noexcept -> CUdevice  { return m_DevCU; }
			private:
				CUcontext m_CtxCU = nullptr;
				CUdevice m_DevCU = 0;
			};
		}
	}
}
#endif
