#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_PROGRAM_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_PROGRAM_H
#include "ImplGLResource.h"
#include <optional>
#include <string>
#include <unordered_map>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLShader;
				class ImplGLProgram : public ImplGLResource {
				public:
					virtual ~ImplGLProgram()noexcept;

					bool AttachShader(ImplGLShader* shader);

					bool Link();
					bool Link(std::string& logData);

					bool IsAvailable()const noexcept;
					bool IsLinkable()const noexcept;
					bool IsAttachable(GLenum shaderType)const noexcept;
				protected:
					ImplGLProgram(ImplGLResourceTable* table)noexcept;
					void AddShaderType(GLenum shaderType, bool isRequired = false)noexcept;
				private:
					struct AttachState {
						bool isRequired = false;
						bool isEnabled  = false;
					};
					struct PrelinkState {
						std::unordered_map<GLenum, AttachState> attachedStages= {};
					};
					std::optional<PrelinkState> m_PrelinkState = PrelinkState{};
				};
				class ImplGLGraphicsProgram: public ImplGLProgram {
				public:
					static auto New(ImplGLResourceTable* table)->ImplGLGraphicsProgram*;
					virtual ~ImplGLGraphicsProgram()noexcept;
				private:
					ImplGLGraphicsProgram(ImplGLResourceTable* table)noexcept;
				};
				class  ImplGLComputeProgram: public ImplGLProgram {
				public:
					static auto New(ImplGLResourceTable* table)->ImplGLComputeProgram*;
					virtual ~ImplGLComputeProgram()noexcept;
				private:
					ImplGLComputeProgram(ImplGLResourceTable* table)noexcept;

				};
			}
		}
	}
}
#endif