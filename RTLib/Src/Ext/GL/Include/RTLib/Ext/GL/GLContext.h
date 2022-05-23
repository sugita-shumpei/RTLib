#ifndef RTLIB_EXT_GL_GL_CONTEXT_H
#define RTLIB_EXT_GL_GL_CONTEXT_H
#include <RTLib/Core/Context.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
#include <vector>
#include <memory>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			class GLBuffer;
			class GLTexture;
			class GLImage;
			class GLProgram;
			class GLShader;
			class GLContextState;
			class GLVertexArray;
			class GLContext : public Core::Context
			{
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLContext, Core::Context, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_CONTEXT);
			public:
				friend class GLBuffer;
				friend class GLImage;
				friend class GLTexture;
				friend class GLShader;
				friend class GLProgram;
				friend class GLVertexArray;
			public:
				GLContext() noexcept;
                virtual ~GLContext()noexcept;

				virtual bool Initialize() override;
				virtual void Terminate() override;
				// Context
				virtual bool InitLoader() = 0;
				virtual void FreeLoader() = 0;
				/*Version*/
				bool SupportVersion(uint32_t majorVersion, uint32_t minorVersion) const noexcept;
				auto GetMajorVersion() const noexcept -> uint32_t;
				auto GetMinorVersion() const noexcept -> uint32_t;
				/*Create*/
				auto CreateBuffer(const GLBufferCreateDesc &desc) -> GLBuffer *;
				auto CreateTexture(const GLTextureCreateDesc &desc) -> GLTexture *;
				auto CreateShader(GLShaderStageFlagBits shaderType) -> GLShader *;
				auto CreateProgram() -> GLProgram *;
				auto CreateVertexArray() -> GLVertexArray *;
				/*Copy*/
				bool CopyBuffer(GLBuffer *srcBuffer, GLBuffer *dstBuffer, const std::vector<GLBufferCopy> &regions);
				bool CopyMemoryToBuffer(GLBuffer *buffer, const std::vector<GLMemoryBufferCopy> &regions);
				bool CopyBufferToMemory(GLBuffer *buffer, const std::vector<GLBufferMemoryCopy> &regions);
				bool CopyImageToBuffer(GLImage *srcImage  , GLBuffer*dstBuffer, const std::vector<GLBufferImageCopy> &regions);
				bool CopyBufferToImage(GLBuffer *srcBuffer, GLImage *dstImage, const std::vector<GLBufferImageCopy> &regions);
				bool CopyImageToMemory(GLImage *image, const std::vector<GLImageMemoryCopy> &regions);
				bool CopyMemoryToImage(GLImage *image, const std::vector<GLImageMemoryCopy> &regions);
                /*PipelineState*/
                void SetProgram(GLProgram* program);
                void SetVertexArrayState(GLVertexArray* vao);
				/*Draw*/
				void DrawArrays(  GLDrawMode drawMode, size_t first, int32_t count);
				void DrawElements(GLDrawMode drawMode, GLIndexFormat indexType, size_t count, intptr_t indexOffsetInBytes);
				/*Clear*/
				void SetClearBuffer(GLClearBufferFlags flags);
				void SetClearColor(float r, float g, float b, float a);
			private:
				auto GetContextState() const noexcept -> const GLContextState *;
				auto GetContextState() noexcept -> GLContextState*;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
