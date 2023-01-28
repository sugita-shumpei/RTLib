#include <RTLib/Ext/CUGL/CUGLBuffer.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
#include <cudaGL.h>
struct RTLib::Ext::CUGL::CUGLBuffer::Impl
{
	Impl(CUDA::CUDAContext* ctx, GL::GLBuffer* buffer)noexcept :ctxCUDA{ ctx }, bufferGL{ buffer }, graphicsResource{ nullptr }, bufferCU{ nullptr }{}
	CUDA::CUDAContext*                ctxCUDA;
	GL::GLBuffer*                     bufferGL;
	std::unique_ptr<CUDA::CUDABuffer> bufferCU;
	CUgraphicsResource                graphicsResource;
	
};
       RTLib::Ext::CUGL::CUGLBuffer::CUGLBuffer(CUDA::CUDAContext* ctx, GL::GLBuffer* bufferGL)noexcept :m_Impl{ new Impl(ctx,bufferGL) }{}
auto   RTLib::Ext::CUGL::CUGLBuffer::New(CUDA::CUDAContext* ctx, GL::GLBuffer* bufferGL, unsigned int flags) -> CUGLBuffer*
{
	auto buffer = new CUGLBuffer(ctx, bufferGL);
	CUgraphicsResource resource;
	RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsGLRegisterBuffer(&resource, GL::GLNatives::GetResId(bufferGL), flags));
	buffer->m_Impl->graphicsResource = resource;
	return buffer;
}
       RTLib::Ext::CUGL::CUGLBuffer::~CUGLBuffer() noexcept
{
	m_Impl.reset();
}
void   RTLib::Ext::CUGL::CUGLBuffer::Destroy() noexcept
{
	if (!m_Impl) { return; }
	if (m_Impl->graphicsResource) {
		try {
			RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsUnregisterResource(m_Impl->graphicsResource));
		}
		catch (CUDA::CUDAException& exception) {
			std::cerr << exception.what() << std::endl;
		}
	}
	m_Impl->ctxCUDA = nullptr;
	m_Impl->bufferGL = nullptr;
	m_Impl->graphicsResource = nullptr;
}

auto RTLib::Ext::CUGL::CUGLBuffer::Map(CUDA::CUDAStream* stream) -> CUDA::CUDABuffer*
{
	if (m_Impl->bufferCU) { return m_Impl->bufferCU.get(); }
	CUdeviceptr deviceptr = 0;
	size_t sizeInBytes = 0;
	auto desc = CUDA::CUDABufferCreateDesc();
	desc.flags = CUDA::CUDAMemoryFlags::eDefault;
	desc.sizeInBytes = sizeInBytes;
	RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsMapResources(1, &m_Impl->graphicsResource, CUDA::CUDANatives::GetCUstream(stream)));
	RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsResourceGetMappedPointer(&deviceptr,&sizeInBytes,m_Impl->graphicsResource));
	m_Impl->bufferCU = std::unique_ptr<CUDA::CUDABuffer>(CUDA::CUDANatives::GetCUDABuffer(m_Impl->ctxCUDA, deviceptr, desc.sizeInBytes,desc.flags));
	return m_Impl->bufferCU.get();
}

void RTLib::Ext::CUGL::CUGLBuffer::Unmap(CUDA::CUDAStream* stream)
{
	if (!m_Impl->bufferCU) { return; }
	m_Impl->bufferCU->Destroy();
	m_Impl->bufferCU.reset();
	RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsUnmapResources(1, &m_Impl->graphicsResource, CUDA::CUDANatives::GetCUstream(stream)));
}

auto RTLib::Ext::CUGL::CUGLBuffer::GetContextCU() noexcept -> CUDA::CUDAContext*
{
	return m_Impl->ctxCUDA;
}

auto RTLib::Ext::CUGL::CUGLBuffer::GetContextCU() const noexcept -> const CUDA::CUDAContext*
{
	return m_Impl->ctxCUDA;
}
