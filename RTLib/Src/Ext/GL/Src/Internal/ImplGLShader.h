#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_SHADER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_SHADER_H
#include "ImplGLResource.h"
#include <glad/glad.h>
#include <string>
#include <vector>
#include <optional>
#include <variant>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				struct ShaderSpecializeConstantInfo {
					uint32_t constantIndex;

				};
				class ImplGLShader : public ImplGLResource {
				public:
					friend class ImplGLShaderModule;
					friend class ImplGLProgram;
					virtual ~ImplGLShader()noexcept;
					//ResetSourceGLSL
					bool ResetSourceGLSL(const std::vector<char>& source);
					bool Compile();
					bool Compile(std::string& logData);
					bool IsAttachable()const noexcept;
					//ResetBinarySpirv
					bool ResetBinarySPV(const std::vector<uint32_t>& spirvData);
					bool Specialize(const GLchar* pEntryPoint​, GLuint numSpecializationConstants​ = 0, const GLuint* pConstantIndex = nullptr, const GLuint* pConstantValue​ = nullptr);
					auto GetShaderType()const noexcept -> GLenum;
				protected:
					ImplGLShader(ImplGLResourceTable* table, bool isSpirvSupported = false)noexcept;
				private:
					struct PreAttachableState {
						bool ownSource = false;
						bool ownBinary = false;
					};
					bool m_SpvSupported = false;
					std::optional< PreAttachableState> m_PreAttachableState = PreAttachableState{};
				};
				class ImplGLVertexShader : public ImplGLShader {
				public:
					static auto New(ImplGLResourceTable* table, bool isSpirvSupported)->ImplGLVertexShader*;
					virtual ~ImplGLVertexShader()noexcept;
				protected:
					ImplGLVertexShader(ImplGLResourceTable* table, bool isSpirvSupported = false)noexcept;
				};
				class ImplGLFragmentShader : public ImplGLShader {
				public:
					static auto New(ImplGLResourceTable* table, bool isSpirvSupported)->ImplGLFragmentShader*;
					virtual ~ImplGLFragmentShader()noexcept;
				protected:
					ImplGLFragmentShader(ImplGLResourceTable* table, bool isSpirvSupported = false)noexcept;
				};
				class ImplGLGeometryShader : public ImplGLShader {
				public:
					static auto New(ImplGLResourceTable* table, bool isSpirvSupported)->ImplGLGeometryShader*;
					virtual ~ImplGLGeometryShader()noexcept;
				protected:
					ImplGLGeometryShader(ImplGLResourceTable* table, bool isSpirvSupported = false)noexcept;
				};
				class ImplGLTesselationEvaluationShader :public  ImplGLShader {
				public:
					static auto New(ImplGLResourceTable* table, bool isSpirvSupported)->ImplGLTesselationEvaluationShader*;
					virtual ~ImplGLTesselationEvaluationShader()noexcept;
				protected:
					ImplGLTesselationEvaluationShader(ImplGLResourceTable* table, bool isSpirvSupported = false)noexcept;
				};
				class ImplGLTesselationControlShader : public ImplGLShader {
				public:
					static auto New(ImplGLResourceTable* table, bool isSpirvSupported)->ImplGLTesselationControlShader*;
					virtual ~ImplGLTesselationControlShader()noexcept;
				protected:
					ImplGLTesselationControlShader(ImplGLResourceTable* table, bool isSpirvSupported = false)noexcept;
				};
				class ImplGLComputeShader : public ImplGLShader {
				public:
					static auto New(ImplGLResourceTable* table, bool isSpirvSupported)->ImplGLComputeShader*;
					virtual ~ImplGLComputeShader()noexcept;
				protected:
					ImplGLComputeShader(ImplGLResourceTable* table, bool isSpirvSupported = false)noexcept;
				};

			}
		}
	}
}
#endif