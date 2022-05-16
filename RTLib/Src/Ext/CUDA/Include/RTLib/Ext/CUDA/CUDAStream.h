#ifndef RTLIB_EXT_CUDA_CUDA_STREAM_H
#define RTLIB_EXT_CUDA_CUDA_STREAM_H
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <RTLib/Core/Context.h>
#include <vector>
namespace RTLib {
	namespace Ext
	{
		namespace CUDA
		{
			class CUDAContext;
			class CUDABuffer;
			class CUDAImage;
			class CUDATexture;
			class CUDAModule;
			class CUDAStream
			{
				friend class CUDAFunction;
			public:
				static auto New(CUDAContext* context)->CUDAStream*;
				virtual ~CUDAStream()noexcept;
				void Destroy();

				bool Synchronize();
				/*Copy*/
				bool CopyBuffer(CUDABuffer* srcBuffer, CUDABuffer* dstBuffer, const std::vector<CUDABufferCopy>& regions);

				bool CopyMemoryToBuffer(CUDABuffer* buffer, const std::vector<CUDAMemoryBufferCopy>& regions);

				bool CopyBufferToMemory(CUDABuffer* buffer, const std::vector<CUDABufferMemoryCopy>& regions);

				bool CopyImageToBuffer(CUDAImage* srcImage, CUDABuffer* dstBuffer, const std::vector<CUDABufferImageCopy>& regions);

				bool CopyBufferToImage(CUDABuffer* srcBuffer, CUDAImage* dstImage, const std::vector<CUDABufferImageCopy>& regions);

				bool CopyImageToMemory(CUDAImage* image, const std::vector<CUDAImageMemoryCopy>& regions);

				bool CopyMemoryToImage(CUDAImage* image, const std::vector<CUDAMemoryImageCopy>& regions);

			private:
				CUDAStream(CUDAContext* context, CUstream stream)noexcept;
				auto GetCUStream()noexcept -> CUstream;
			private:
				CUDAContext* m_Context = nullptr;
				CUstream     m_Stream  = nullptr;
			};
		}
	}
}
#endif
