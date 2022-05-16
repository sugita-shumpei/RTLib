#ifndef RTLIB_EXT_GL_GL_CONTEXT_STATE_H
#define RTLIB_EXT_GL_GL_CONTEXT_STATE_H
#include <RTLib/Ext/GL/GLCommon.h>
#include "GLBaseObject.h"
#include <vector>
namespace RTLib
{
	namespace Ext {
		namespace GL
		{
			class GLContextState
			{
			public:
				void SetProgram (const GLBaseProgram& program);
				void SetProgramPipeline(const GLBaseProgramPipeline& programPipeline);
				void BindBuffer (GLenum target, const GLBaseBuffer& obj, bool forVaoReset);
				void BindUniformBuffer(int32_t index, const GLBaseBuffer& obj, GLintptr offset, GLsizeiptr size);
				void BindStorageBuffer(int32_t index, const GLBaseBuffer& obj, GLintptr offset, GLsizeiptr size);
				void SetActiveTexture(int32_t index);
				void BindTexture(int32_t index, GLenum target, const GLBaseTexture& tex);
				void BindSampler(uint32_t index, const GLBaseSampler& smp);

				void BindVertexArray( const GLBaseVertexArray& vao);
				void BindFramebuffer( const GLBaseFramebuffer& fbo);
				void BindRenderbuffer(const GLBaseRenderbuffer& rbo);
			private:
				std::vector<GLuint> 
			};
		}
	}
}
#endif