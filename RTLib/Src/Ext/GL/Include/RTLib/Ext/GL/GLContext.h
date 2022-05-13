#ifndef RTLIB_EXT_GL_GL_CONTEXT_H
#define RTLIB_EXT_GL_GL_CONTEXT_H
#include <RTLib/Core/Context.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <vector>
#include <memory>
namespace RTLib {
	namespace Ext {
		namespace GL {
			class GLBuffer  ;
			class GLImage   ;
			class GLTexture ;
			class GLContext : public Core::Context
			{
			public:
				GLContext()noexcept;
				virtual ~GLContext()noexcept;

				virtual bool Initialize() override;
				virtual void Terminate () override;
				// Context ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
				virtual bool InitLoader() = 0;
				virtual void FreeLoader() = 0;

				virtual auto CreateBuffer(const  GLBufferDesc & desc)-> GLBuffer * = 0;
				virtual auto CreateTexture(const GLTextureDesc& desc)-> GLTexture* = 0;
				/*Copy*/
				bool CopyBuffer(GLBuffer* srcBuffer, GLBuffer* dstBuffer, const std::vector<GLBufferCopy>& regions);

				bool CopyMemoryToBuffer(GLBuffer* buffer, const std::vector<GLMemoryBufferCopy>& regions);

				bool CopyBufferToMemory(GLBuffer* buffer, const std::vector<GLBufferMemoryCopy>& regions);

				bool CopyImageToBuffer(GLImage*  srcImage , GLBuffer* dstBuffer, const std::vector<GLBufferImageCopy>& regions);

				bool CopyBufferToImage(GLBuffer* srcBuffer, GLImage*   dstImage, const std::vector<GLBufferImageCopy>& regions);

				bool CopyImageToMemory(GLImage* image, const std::vector<GLImageMemoryCopy>& regions);

				bool CopyMemoryToImage(GLImage* image, const std::vector<GLImageMemoryCopy>& regions);
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
