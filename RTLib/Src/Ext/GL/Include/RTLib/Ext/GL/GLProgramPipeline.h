#ifndef RTLIB_EXT_GL_GL_PROGRAM_PIPELINE_H
#define RTLIB_EXT_GL_GL_PROGRAM_PIPELINE_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
#include <unordered_map>
#include <memory>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			class GLContext;
			class GLProgram;
			class GLProgramPipeline : public Core::BaseObject
			{
				friend class GLProgram;
				friend class GLNatives;
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLProgramPipeline, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_PROGRAM_PIPELINE);
			public:
				static auto New(GLContext* context)->GLProgramPipeline*;
				virtual ~GLProgramPipeline()noexcept;
				void Destroy();

				auto GetProgram(GLShaderStageFlagBits stage)const noexcept->const GLProgram*;
				auto GetProgram(GLShaderStageFlagBits stage)noexcept->GLProgram*;
				void SetProgram(GLShaderStageFlags   stages, GLProgram* program);
			private:
				GLProgramPipeline(GLContext* context)noexcept;
				auto GetResId()const noexcept -> GLuint;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};

		}
	}
}
#endif
