#ifndef RTLIB_EXT_GL_GL_OBJECT_BASE_H
#define RTLIB_EXT_GL_GL_OBJECT_BASE_H
#include <RTLib/Ext/GL/GLCommon.h>
#include <cstdint>
#include "GLUniqueID.h"
namespace RTLib { 
	namespace Ext {
		namespace GL {
			template<typename GLCreateDeleter>
			class  GLObjectBase {
			public:
				GLObjectBase(bool isCreate, GLCreateDeleter createDeleter = GLCreateDeleter())noexcept:m_UniqueID(), m_CreateDeleter{createDeleter}
				{
					if (isCreate) {
						Create();
					}
				}

				GLObjectBase(const GLObjectBase&) = delete;
				GLObjectBase& operator=(const GLObjectBase&) = delete;

				GLObjectBase(GLObjectBase&& objectBase)noexcept {
					m_ObjectID = std::exchange(objectBase.m_ObjectID, 0);
					m_UniqueID = std::move(objectBase.m_UniqueID);
					m_CreateDeleter = std::move(objectBase.m_CreateDeleter);
				}
				GLObjectBase& operator=(GLObjectBase&& objectBase)noexcept {
					if (this != &objectBase) {
						m_ObjectID = std::exchange(objectBase.m_ObjectID, 0);
						m_UniqueID = std::move(objectBase.m_UniqueID);
						m_CreateDeleter = std::move(objectBase.m_CreateDeleter);
					}
					return *this;
				}
				
				static auto Null() ->GLObjectBase {
					return GLObjectBase(false);
				}

				void Create()noexcept  {
					Destroy();
					m_ObjectID = m_CreateDeleter.Create();
				}
				void Destroy()noexcept {
					if (m_ObjectID == 0) { return; }
					m_CreateDeleter.Destroy();
					m_ObjectID = 0;
				}

				explicit operator GLuint ()const noexcept { return m_ObjectID; }

				auto GetObjectID()const noexcept -> GLuint     { return m_ObjectID; }
				auto GetUniqueID()const noexcept -> GLUniqueId { return m_UniqueID.GetID(); }
			private:
				GLuint           m_ObjectID;
				GLUniqueIdHolder m_UniqueID;
				GLCreateDeleter  m_CreateDeleter;
			};
			struct GLCreateDeleterBuffer 
			{
				GLCreateDeleterBuffer(GLuint externalObjectId = 0)noexcept:m_ExternalObjectId{externalObjectId}{}
				auto Create()->GLuint {
					if (m_ExternalObjectId) { return m_ExternalObjectId; }
					GLuint objectID = 0;
					glGenBuffers(1, &objectID);
					return objectID;
				}
				void Destroy(GLuint objectID) {
					if (m_ExternalObjectId) { m_ExternalObjectId = 0; return; }
					glDeleteBuffers(1, &objectID);
				}
			private:
				GLuint m_ExternalObjectId = 0;
			};
			struct GLCreateDeleterTexture
			{
				GLCreateDeleterTexture(GLuint externalObjectId = 0)noexcept :m_ExternalObjectId{ externalObjectId } {}
				auto Create()->GLuint {
					if (m_ExternalObjectId) { return m_ExternalObjectId; }
					GLuint objectID = 0;
					glGenTextures(1, &objectID);
					return objectID;
				}
				void Destroy(GLuint objectID) {
					if (m_ExternalObjectId) { m_ExternalObjectId = 0; return; }
					glDeleteTextures(1, &objectID);
				}
			private:
				GLuint m_ExternalObjectId = 0;
			};
			struct GLCreateDeleterSampler
			{
				GLCreateDeleterSampler(GLuint externalObjectId)noexcept{}
				auto Create()->GLuint {
					GLuint objectID = 0;
					glGenSamplers(1, &objectID);
					return objectID;
				}
				void Destroy(GLuint objectID) {
					glDeleteSamplers(1, &objectID);
				}
			private:
				GLuint m_ExternalObjectId = 0;
			};
			struct GLCreateDeleterQuery
			{
				GLCreateDeleterQuery()noexcept {}
				auto Create()->GLuint {
					GLuint objectID = 0;
					glGenQueries(1, &objectID);
					return objectID;
				}
				void Destroy(GLuint objectID) {
					glDeleteQueries(1, &objectID);
				}
			};
			struct GLCreateDeleterShader  {
				GLCreateDeleterShader(uint32_t shaderType)noexcept :m_ShaderType{shaderType} {}
				auto Create()->GLuint {
					return glCreateShader(m_ShaderType);
				}
				void Destroy(GLuint objectID) {
					glDeleteShader(objectID);
				}
			private:
				uint32_t m_ShaderType;
			};
			struct GLCreateDeleterProgram {
				GLCreateDeleterProgram()noexcept {}
				auto Create()->GLuint {
					return glCreateProgram();
				}
				void Destroy(GLuint objectID) {
					glDeleteProgram(objectID);
				}
			};
			struct GLCreateDeleterProgramPipeline {
				GLCreateDeleterProgramPipeline()noexcept {}
				auto Create()->GLuint {
					GLuint objectID = 0;
					glGenProgramPipelines(1, &objectID);
					return objectID;
				}
				void Destroy(GLuint objectID) {
					glDeleteProgramPipelines(1, &objectID);
				}
			};
			struct GLCreateDeleterVAO {
				GLCreateDeleterVAO()noexcept {}
				auto Create()->GLuint {
					GLuint objectID = 0;
					glGenVertexArrays(1, &objectID);
					return objectID;
				}
				void Destroy(GLuint objectID) {
					glDeleteVertexArrays(1, &objectID);
				}
			};
			struct GLCreateDeleterFBO {
				GLCreateDeleterFBO()noexcept {}
				auto Create()->GLuint {
					GLuint objectID = 0;
					glGenFramebuffers(1, &objectID);
					return objectID;
				}
				void Destroy(GLuint objectID) {
					glDeleteFramebuffers(1, &objectID);
				}
			};
			struct GLCreateDeleterRBO {
				GLCreateDeleterRBO()noexcept {}
				auto Create()->GLuint {
					GLuint objectID = 0;
					glGenRenderbuffers(1, &objectID);
					return objectID;
				}
				void Destroy(GLuint objectID) {
					glDeleteRenderbuffers(1, &objectID);
				}
			};

			using  GLBaseBuffer			 = GLObjectBase<GLCreateDeleterBuffer>;
			using  GLBaseTexture		 = GLObjectBase<GLCreateDeleterTexture>;
			using  GLBaseSampler		 = GLObjectBase<GLCreateDeleterSampler>;
			using  GLBaseQuery			 = GLObjectBase<GLCreateDeleterQuery>;
			using  GLBaseShader			 = GLObjectBase<GLCreateDeleterShader>;
			using  GLBaseProgram	     = GLObjectBase<GLCreateDeleterProgram>;
			using  GLBaseProgramPipeline = GLObjectBase<GLCreateDeleterProgramPipeline>;
			using  GLBaseVAO  		     = GLObjectBase<GLCreateDeleterVAO>;
			using  GLBaseFBO		     = GLObjectBase<GLCreateDeleterFBO>;
			using  GLBaseRBO  		     = GLObjectBase<GLCreateDeleterRBO>;
		} 
	} 
}
#endif
