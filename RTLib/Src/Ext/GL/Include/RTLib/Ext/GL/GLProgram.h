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
			RTLIB_CORE_TYPE_OBJECT_DECLARE_BEGIN(GLProgram, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_PROGRAM);

		public:
			friend class GLContext;
			friend class GLProgramPipeline;
			friend class GLShader;
			static auto New(GLContext* context)->GLProgram*;
			void Destroy();
			bool AttachShader(GLShader *shader);

			bool Link();
			bool Link(std::string &logData);

			bool IsLinked() const noexcept;
			bool IsLinkable() const noexcept;
			bool IsAttachable(GLenum shaderType) const noexcept;
			// USE
			bool Enable();
			void Disable();
			bool IsEnabled() const noexcept;
			//
			bool HasShaderType(GLenum shaderType) const noexcept;
			auto GetShaderStages() const noexcept -> GLbitfield;

			auto GetUniformBlockIndex(const char *name) -> GLuint;
			bool SetUniformBlockBinding(GLuint blockIndex, GLuint bindingIndex);

		protected:
			GLProgram(GLContext* context) noexcept;
			void AddShaderType(GLenum shaderType, bool isRequired = false) noexcept;
			auto GetResId()const noexcept ->GLuint;
		private:
			struct AttachState
			{
				bool isRequired = false;
				bool isEnabled  = false;
			};
			GLContext* m_Context = nullptr;
			std::unordered_map<GLenum, AttachState> m_AttachedStages = {};
			bool m_IsEnabled = false;
			bool m_IsLinked = false;
			GLuint m_ResId = 0;
		RTLIB_CORE_TYPE_OBJECT_DECLARE_END();
		}
	}
}
#endif
