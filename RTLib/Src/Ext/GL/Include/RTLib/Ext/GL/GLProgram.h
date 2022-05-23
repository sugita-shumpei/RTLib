#ifndef RTLIB_EXT_GL_GL_PROGRAM_H
#define RTLIB_EXT_GL_GL_PROGRAM_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <unordered_map>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			class GLContext;
			class GLShader;
			class GLProgram : public Core::BaseObject
			{
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLProgram, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_PROGRAM);

			public:
				friend class GLContext;
				friend class GLProgramPipeline;
				friend class GLShader;
				static auto New(GLContext *context) -> GLProgram *;
                virtual ~GLProgram()noexcept;
                
				void Destroy();
				bool AttachShader(GLShader *shader);

				bool Link();
				bool Link(std::string &logData);

				bool IsLinked() const noexcept;
				bool IsLinkable() const noexcept;
				bool IsAttachable(GLShaderStageFlagBits shaderType) const noexcept;
				// USE
				bool Enable();
				void Disable();
				bool IsEnabled() const noexcept;
				//
				bool HasShaderType(GLShaderStageFlagBits shaderType) const noexcept;
				auto GetShaderStages() const noexcept -> GLbitfield;

				auto GetUniformLocation(const char* name)->GLint;

				auto GetUniformBlockIndex(const char *name) -> GLuint;
				bool SetUniformBlockBinding(GLuint blockIndex, GLuint bindingIndex);

				bool SetUniformImageUnit(GLint location, GLuint imageUnit);
			protected:
				GLProgram(GLContext *context) noexcept;
				void AddShaderType(GLShaderStageFlagBits shaderType, bool isRequired = false) noexcept;
				auto GetResId() const noexcept -> GLuint;
			private:
				void SetImageUnit(GLint location, GLint imageUnit)noexcept;
			private:
				struct AttachState
				{
					bool isRequired = false;
					bool isEnabled = false;
				};
				GLContext *m_Context = nullptr;
				std::unordered_map<GLenum, AttachState> m_AttachedStages = {};
				std::unordered_map<GLint , GLint>       m_ImageUnits     = {};
				bool m_IsEnabled = false;
				bool m_IsLinked = false;
				GLuint m_ResId = 0;
			};
		}
	}
}
#endif
