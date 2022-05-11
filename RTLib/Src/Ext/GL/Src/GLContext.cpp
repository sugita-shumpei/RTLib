#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLBuffer.h>

bool RTLib::Ext::GL::GLContext::CopyBuffer(GLBuffer* srcBuffer, GLBuffer* dstBuffer, const std::vector<GLBufferCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyMemoryToBuffer(GLBuffer* buffer, const std::vector<GLMemoryBufferCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyBufferToMemory(GLBuffer* buffer, const std::vector<GLBufferMemoryCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyImageToBuffer(GLImage* srcImage, GLBuffer* dstBuffer, const std::vector<GLBufferImageCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyBufferToImage(GLBuffer* srcBuffer, GLImage* dstImage, const std::vector<GLBufferImageCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyImageToMemory(GLImage* image, const std::vector<GLImageMemoryCopy>& regions)
{
	return false;
}

bool RTLib::Ext::GL::GLContext::CopyMemoryToImage(GLImage* image, const std::vector<GLImageMemoryCopy>& regions)
{
	return false;
}
