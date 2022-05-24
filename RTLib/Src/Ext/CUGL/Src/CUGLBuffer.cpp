#include <RTLib/Ext/CUGL/CUGLBuffer.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
#include <cudaGL.h>
struct RTLib::Ext::CUGL::CUGLBuffer::Impl
{
	Impl(CUGLContext* ctx, GL::GLBuffer* buffer)noexcept :ctxCUGL{ ctx }, bufferGL{ buffer }, graphicsResource{ nullptr }, isMapped{ false }{}
	CUGL::CUGLContext* ctxCUGL;
	GL::GLBuffer*      bufferGL;
	CUgraphicsResource graphicsResource;
	bool               isMapped;
	
};
       RTLib::Ext::CUGL::CUGLBuffer::CUGLBuffer(CUGLContext* ctx, GL::GLBuffer* bufferGL)noexcept :m_Impl{ new Impl(ctx,bufferGL) }{}
auto   RTLib::Ext::CUGL::CUGLBuffer::New(CUGLContext* ctx, GL::GLBuffer* bufferGL, unsigned int flags) -> CUGLBuffer*
{
	auto buffer = new CUGLBuffer(ctx, bufferGL);
	CUgraphicsResource resource;
	RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsGLRegisterBuffer(&resource, 0, flags));
	buffer->m_Impl->graphicsResource = resource;
	return buffer;
}
       RTLib::Ext::CUGL::CUGLBuffer::~CUGLBuffer() noexcept
{
	
}
void   RTLib::Ext::CUGL::CUGLBuffer::Destroy() noexcept
{
	if (!m_Impl) { return; }
	if (m_Impl->graphicsResource) {
		RTLIB_EXT_CUDA_THROW_IF_FAILED(cuGraphicsUnregisterResource(m_Impl->graphicsResource));
	}
	m_Impl->ctxCUGL = nullptr;
	m_Impl->bufferGL = nullptr;
	m_Impl->graphicsResource = nullptr;
}

