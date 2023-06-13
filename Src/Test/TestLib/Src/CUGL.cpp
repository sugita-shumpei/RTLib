#include <TestLib/CUGL.h>
#include <OptiXToolkit/Error/ErrorCheck.h>
TestLib::BufferCUGL::BufferCUGL(GLuint id, unsigned int graphicsResourceFlags)
	:m_ID{id}
{
	cuGraphicsGLRegisterBuffer(&m_GraphicsResource, id, graphicsResourceFlags);
}

TestLib::BufferCUGL::~BufferCUGL()
{
	OTK_ERROR_CHECK(cuGraphicsUnregisterResource(m_GraphicsResource));
	m_GraphicsResource = nullptr;
}

auto TestLib::BufferCUGL::map(CUstream stream, size_t& size) -> CUdeviceptr
{
	CUdeviceptr res = 0;
	OTK_ERROR_CHECK(cuGraphicsMapResources(1, &m_GraphicsResource, stream));
	OTK_ERROR_CHECK(cuGraphicsResourceGetMappedPointer(&res, &size, m_GraphicsResource));
	return res;
}

void TestLib::BufferCUGL::unmap(CUstream stream)
{
	OTK_ERROR_CHECK(cuGraphicsUnmapResources(1, &m_GraphicsResource, stream));
}


auto TestLib::BufferCUGL::get_id() const noexcept -> GLuint { return m_ID; }