#ifndef TEST_TESTLIB_CUGL__H
#define TEST_TESTLIB_CUGL__H
#include <glad/gl.h>
#include <cuda.h>
#include <cudaGL.h>
namespace TestLib
{
	struct BufferCUGL
	{
		 BufferCUGL(GLuint id,unsigned int graphicsResourceFlags);
		~BufferCUGL();

		auto map(CUstream stream, size_t& size) -> CUdeviceptr;
		void unmap(CUstream stream);
		
		auto get_id() const noexcept -> GLuint;
	private:
		GLuint             m_ID;
		CUgraphicsResource m_GraphicsResource;
	};
}

#endif
