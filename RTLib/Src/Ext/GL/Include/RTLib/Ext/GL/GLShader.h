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
			class GLShader :public Core::BaseObject
			{
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLShader, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_SHADER);
			public:
				friend class GLContext;
				friend class GLProgram;
				static auto New(GLContext *context, GLShaderStageFlagBits shaderStage, bool isSpirvSupport) -> GLShader *;
                virtual ~GLShader()noexcept;
                
				void Destroy();
				// ResetSourceGLSL
				bool ResetSourceGLSL(const std::vector<char> &source);
				bool Compile();
				bool Compile(std::string &logData);
				bool IsAttachable() const noexcept;
				// ResetBinarySpirv
				bool ResetBinarySPV(const std::vector<uint32_t> &spirvData);
				bool Specialize(const GLchar *pEntryPoint​,GLuint numPushConstants = 0, const GLuint *pConstantIndex = nullptr, const GLuint *pConstantValue​ = nullptr);
				auto GetShaderStage() const noexcept -> GLShaderStageFlagBits;
			protected:
				GLShader(GLContext *context, GLShaderStageFlagBits shaderStage, bool isSpirvSupported = false) noexcept;
				GLuint GetResId() const noexcept;

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
			};
		}
	}
}
#endif
