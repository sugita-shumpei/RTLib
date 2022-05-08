#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_SHADER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_SHADER_H
#include "ImplGLResource.h"
#include <glad/glad.h>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLShaderBase : public ImplGLResourceBase {
				public:
					ImplGLShaderBase(GLenum shaderType):ImplGLResourceBase(), m_ShaderType{shaderType}{}
					virtual ~ImplGLShaderBase()noexcept {}

					auto GetShaderType()const noexcept -> GLenum { return m_ShaderType; }
				protected:
					virtual bool  Create()noexcept override {
						GLuint resId = glCreateShader(m_ShaderType);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void Destroy()noexcept override {
						glDeleteShader(GetResId());
						SetResId(0);
					}
				private:
					GLenum m_ShaderType;
				};
				class ImplGLShader : public ImplGLResource {
				public:
					static auto New(ImplGLResourceTable* table, GLenum shaderType)->ImplGLShader* {
						if (!table) {
							return nullptr;
						}
						auto program = new ImplGLShader(table);
						if (program) {
							program->InitBase<ImplGLShaderBase>(shaderType);
							auto res = program->Create();
							if (!res) {
								delete program;
								return nullptr;
							}
						}
						return program;
					}
					virtual ~ImplGLShader()noexcept {}

					auto GetShaderType()const noexcept -> GLenum {
						auto base = GetBase();
						if (base) {
							return static_cast<const ImplGLShaderBase*>(base)->GetShaderType();
						}
						else {
							return GL_COMPUTE_SHADER;
						}
					}
				protected:
					ImplGLShader(ImplGLResourceTable* table)noexcept :ImplGLResource(table) {}
				private:
					bool m_InitSource;
				};
			}
		}
	}
}
#endif