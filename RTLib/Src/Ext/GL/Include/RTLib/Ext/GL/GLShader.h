#ifndef RTLIB_EXT_GL_GL_SHADER_H
#define RTLIB_EXT_GL_GL_SHADER_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <optional>
#include <vector>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
        class GLContext;
			RTLIB_CORE_TYPE_OBJECT_DECLARE_BEGIN(GLShader, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_SHADER);
		public:
			friend class GLContext;
			friend class GLProgram;
			static auto New(GLContext* context, GLShaderStageFlagBits shaderStage,bool isSpirvSupport)->GLShader*;
			void Destroy();
			// ResetSourceGLSL
			bool ResetSourceGLSL(const std::vector<char> &source);
			bool Compile();
			bool Compile(std::string &logData);
			bool IsAttachable() const noexcept;
			// ResetBinarySpirv
			bool ResetBinarySPV(const std::vector<uint32_t> &spirvData);
			bool Specialize(const GLchar *pEntryPoint​, GLuint numSpecializationConstants​ = 0, const GLuint *pConstantIndex = nullptr, const GLuint *pConstantValue​ = nullptr);
			auto GetShaderType() const noexcept -> GLenum;
		protected:
			GLShader(GLContext* context, GLShaderStageFlagBits shaderStage, bool isSpirvSupported = false) noexcept;
			auto GetResId()const noexcept ->GLuint;
		private:
			struct PreAttachableState
			{
				bool ownSource = false;
				bool ownBinary = false;
			};
			GLShaderStageFlagBits m_ShaderStage = GLShaderStageVertex;
			GLuint m_ResId = 0;
			bool m_SpvSupported = false;
			std::optional<PreAttachableState> m_PreAttachableState = PreAttachableState{};
			RTLIB_CORE_TYPE_OBJECT_DECLARE_END();

		}
	}
}
#endif
