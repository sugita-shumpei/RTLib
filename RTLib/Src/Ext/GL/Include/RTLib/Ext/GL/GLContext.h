#ifndef RTLIB_EXT_GL_GL_CONTEXT_H
#define RTLIB_EXT_GL_GL_CONTEXT_H
#include <RTLib/Core/Context.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
#include <vector>
#include <memory>
namespace RTLib {
	namespace Ext {
		namespace GL {
			class GLContextState;
			RTLIB_CORE_TYPE_OBJECT_DECLARE_BEGIN(GLContext, Core::Context, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_CONTEXT);
				friend class GLBuffer;
				friend class GLImage;
				friend class GLTexture;
				friend class GLShader;
				friend class GLProgram;
			public:
				GLContext()noexcept;

				virtual bool Initialize() override;
				virtual void Terminate () override;
				// Context ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
				virtual bool InitLoader() = 0;
				virtual void FreeLoader() = 0;

				auto CreateBuffer(const  GLBufferCreateDesc & desc)-> GLBuffer * ;
				auto CreateTexture(const GLTextureCreateDesc& desc)-> GLTexture*;
				auto CreateShader(GLShaderStageFlagBits shaderType)-> GLShader *;
				auto CreateProgram()->GLProgram*;
				
				bool SupportVersion(uint32_t majorVersion, uint32_t minorVersion)const noexcept;

				auto GetMajorVersion()const noexcept -> uint32_t;
				auto GetMinorVersion()const noexcept -> uint32_t;
				/*Copy*/
				bool CopyBuffer(GLBuffer* srcBuffer, GLBuffer* dstBuffer, const std::vector<GLBufferCopy>& regions);

				bool CopyMemoryToBuffer(GLBuffer* buffer, const std::vector<GLMemoryBufferCopy>& regions);

				bool CopyBufferToMemory(GLBuffer* buffer, const std::vector<GLBufferMemoryCopy>& regions);

				bool CopyImageToBuffer(GLImage*  srcImage , GLBuffer* dstBuffer, const std::vector<GLBufferImageCopy>& regions);

				bool CopyBufferToImage(GLBuffer* srcBuffer, GLImage*   dstImage, const std::vector<GLBufferImageCopy>& regions);

				bool CopyImageToMemory(GLImage* image, const std::vector<GLImageMemoryCopy>& regions);

				bool CopyMemoryToImage(GLImage* image, const std::vector<GLImageMemoryCopy>& regions);
			private:
				auto GetContextState()const noexcept -> const GLContextState*;
				auto GetContextState()      noexcept ->       GLContextState*;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			RTLIB_CORE_TYPE_OBJECT_DECLARE_END();
		}
	}
}
#endif
