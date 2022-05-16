#ifndef RTLIB_EXT_GL_GL_BASE_OBJECT_H
#define RTLIB_EXT_GL_GL_BASE_OBJECT_H
#include <RTLib/Ext/GL/GLCommon.h>
#include <optional>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{

			template<typename GLCreateDeleteType>
			class GLBaseObject
			{
			public:
				GLBaseObject(bool createObj, GLCreateDeleteType createDeleter = GLCreateDeleteType())noexcept 
					: m_CreateDeleter{ createDeleter }, m_ResId{ 0 } {
					if (createObj) {
						Create();
					}
				}
				~GLBaseObject() {
					Destroy();
				}

				GLBaseObject(const GLBaseObject& obj) = delete;
				GLBaseObject& operator=(const GLBaseObject& obj) = delete;

				GLBaseObject(GLBaseObject&& obj) noexcept {
					m_ResId = obj.m_ResId;
					m_CreateDeleter = std::move(obj.m_CreateDeleter);
				}
				GLBaseObject& operator=(GLBaseObject&& obj) noexcept
				{
					Destroy();
					m_ResId = obj.m_ResId;
					obj.m_ResId = 0;
					m_CreateDeleter = std::move(obj.m_CreateDeleter);
					return *this;
				}

				void Create() {
					Destroy();
					m_ResId = m_CreateDeleter.Create();
				}
				void Destroy()
				{
					if (!m_ResId) { return; }
					m_CreateDeleter.Destroy(m_ResId);
					m_ResId = 0;
				}

				explicit operator GLuint ()const noexcept { return m_ResId; }
				explicit operator bool()const noexcept { return m_ResId != 0; }
			private:
				GLuint             m_ResId;
				GLCreateDeleteType m_CreateDeleter;
			};
			class GLCreateDeleteBuffer  {
			public:
				GLCreateDeleteBuffer(GLuint externalGLResource = 0)noexcept:m_ExternalGLResource{externalGLResource}{}
				auto  Create()->GLuint { 
					if (m_ExternalGLResource!=0) { return m_ExternalGLResource; }
					GLuint resId = 0;  glGenBuffers(1, &resId); return resId; 
				}
				void Destroy(GLuint  resId) {
					if (m_ExternalGLResource!=0) { m_ExternalGLResource = 0; }
					glDeleteBuffers(1, &resId); 
				}
			private:
				GLuint m_ExternalGLResource;
			};
			class GLCreateDeleteTexture {
			public:
				auto  Create()->GLuint {
					if (m_ExternalGLResource != 0) { return m_ExternalGLResource; }
					GLuint resId = 0;  glGenTextures(1, &resId); return resId;
				}
				void Destroy(GLuint  resId) {
					if (m_ExternalGLResource != 0) { m_ExternalGLResource = 0; }
					glDeleteTextures(1, &resId);
				}
			private:
				GLuint m_ExternalGLResource;
			};
			class GLCreateDeleteSampler {
			public:
				auto  Create()->GLuint {
					GLuint resId = 0;  glGenSamplers(1, &resId); return resId;
				}
				void Destroy(GLuint  resId) {
					glDeleteSamplers(1, &resId);
				}
			};
			class GLCreateDeleteFramebuffer {
			public:
				auto  Create()->GLuint { GLuint resId = 0;  glGenFramebuffers(1, &resId); return resId; }
				void Destroy(GLuint  resId) { glDeleteFramebuffers(1, &resId); }
			};
			class GLCreateDeleteRenderbuffer {
			public:
				auto  Create()->GLuint { GLuint resId = 0;  glGenRenderbuffers(1, &resId); return resId; }
				void Destroy(GLuint  resId) { glDeleteRenderbuffers(1, &resId); }
			};
			class GLCreateDeleteVertexArray {
			public:
				auto  Create()->GLuint { GLuint resId = 0;  glGenVertexArrays(1, &resId); return resId; }
				void Destroy(GLuint  resId) { glDeleteVertexArrays(1, &resId); }
			};
			class GLCreateDeleteShader {
			public:
				GLCreateDeleteShader(GLenum shaderType)noexcept :m_ShaderType{ shaderType } {}
				auto  Create()->GLuint { GLuint resId = glCreateShader(m_ShaderType); return resId; }
				void Destroy(GLuint  resId) { glDeleteShader(resId); }
			private:
				GLenum m_ShaderType;
			};
			class GLCreateDeleteProgram {
			public:
				auto  Create()->GLuint { GLuint resId = glCreateProgram(); return resId; }
				void Destroy(GLuint  resId) { glDeleteProgram(resId); }
			};
			class GLCreateDeleteProgramPipeline {
			public:
				auto  Create()->GLuint { GLuint resId = 0; glGenProgramPipelines(1, &resId); return resId; }
				void Destroy(GLuint  resId) { glDeleteProgramPipelines(1,&resId); }
			};
			class GLCreateDeleteQuery {
			public:
				auto  Create()->GLuint { GLuint resId = 0; glGenQueries(1, &resId); return resId; }
				void Destroy(GLuint  resId) { glDeleteQueries(1, &resId); }
			};
			using GLBaseBuffer          = GLBaseObject<GLCreateDeleteBuffer>;
			using GLBaseTexture         = GLBaseObject<GLCreateDeleteTexture>;
			using GLBaseSampler         = GLBaseObject<GLCreateDeleteSampler>;
			using GLBaseFramebuffer     = GLBaseObject<GLCreateDeleteFramebuffer>;
			using GLBaseRenderbuffer    = GLBaseObject<GLCreateDeleteRenderbuffer>;
			using GLBaseVertexArray     = GLBaseObject<GLCreateDeleteVertexArray>;
			using GLBaseShader          = GLBaseObject<GLCreateDeleteShader>;
			using GLBaseProgram			= GLBaseObject<GLCreateDeleteProgram>;
			using GLBaseProgramPipeline = GLBaseObject<GLCreateDeleteProgramPipeline>;
			using GLBaseProgramQuery    = GLBaseObject<GLCreateDeleteQuery>;
		}
	}
}
#endif
