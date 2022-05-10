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
				class ImplGLProgram;
				class ImplGLProgramSlot {
				public:
					friend class ImplGLProgram;
					friend class ImplGLProgramPipeline;
				public:
					 ImplGLProgramSlot()noexcept {}
					~ImplGLProgramSlot()noexcept {}
					bool IsActive()const noexcept;
				private:
					bool Register(ImplGLProgram* program);
					void Unregister();
				private:
					ImplGLProgram* m_UsedProgram = nullptr;
				};
				class ImplGLProgram : public ImplGLResource {
				public:
					friend class ImplGLProgramPipeline;
				public:
					virtual ~ImplGLProgram()noexcept;

					bool AttachShader(ImplGLShader* shader);

					bool Link();
					bool Link(std::string& logData);

					bool IsLinked()const noexcept;
					bool IsLinkable()const noexcept;
					bool IsAttachable(GLenum shaderType)const noexcept;
					//USE
					bool    Enable();
					void   Disable();
					bool IsEnabled()const noexcept;
					//
					bool HasShaderType(GLenum shaderType)const noexcept;
					auto GetShaderStages()const noexcept -> GLbitfield;
				protected:
					ImplGLProgram(ImplGLResourceTable* table, ImplGLProgramSlot* slot)noexcept;
					void AddShaderType(GLenum shaderType, bool isRequired = false)noexcept;
				private:
					struct  AttachState {
						bool isRequired = false;
						bool isEnabled  = false;
					};
					std::unordered_map<GLenum, AttachState> m_AttachedStages = {};
					ImplGLProgramSlot*                      m_ProgramSlot    = nullptr;
					bool                                    m_IsEnabled      = false;
					bool                                    m_IsLinked       = false;
				};
				class ImplGLGraphicsProgram: public ImplGLProgram {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLProgramSlot* slot)->ImplGLGraphicsProgram*;
					virtual ~ImplGLGraphicsProgram()noexcept;
				private:
					ImplGLGraphicsProgram(ImplGLResourceTable* table, ImplGLProgramSlot* slot)noexcept;
				};
				class ImplGLComputeProgram: public ImplGLProgram {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLProgramSlot* slot)->ImplGLComputeProgram*;
					virtual ~ImplGLComputeProgram()noexcept;

					bool Launch(GLuint numGroupX, GLuint numGroupY, GLuint numGroupZ);
				private:
					ImplGLComputeProgram(ImplGLResourceTable* table, ImplGLProgramSlot* slot)noexcept;
				};
				class ImplGLSeparateProgram: public ImplGLProgram {
				public:
					static auto New(ImplGLResourceTable* table, ImplGLProgramSlot* slot)->ImplGLSeparateProgram*;
					virtual ~ImplGLSeparateProgram()noexcept;
				private:
					ImplGLSeparateProgram(ImplGLResourceTable* table, ImplGLProgramSlot* slot)noexcept;
					void SetSeparatable(bool value);
				};
			}
		}
	}
}
#endif