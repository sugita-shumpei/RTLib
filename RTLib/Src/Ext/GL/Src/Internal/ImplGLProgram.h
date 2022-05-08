#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_PROGRAM_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_PROGRAM_H
#include "ImplGLResource.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLProgramBase : public ImplGLResourceBase {
				public:
					virtual ~ImplGLProgramBase()noexcept {}
				protected:
					virtual bool  Create()noexcept override {
						GLuint resId = glCreateProgram();
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void Destroy()noexcept override {
						glDeleteProgram(GetResId());
						SetResId(0);
					}
				};
				class ImplGLProgram : public ImplGLResource {
				public:
					static auto New(ImplGLResourceTable* table)->ImplGLProgram* {
						if (!table) {
							return nullptr;
						}
						auto program = new ImplGLProgram(table);
						if (program) {
							program->InitBase<ImplGLProgramBase>();
							auto res = program->Create();
							if (!res) {
								delete program;
								return nullptr;
							}
						}
						return program;
					}
					virtual ~ImplGLProgram()noexcept {}
				protected:
					ImplGLProgram(ImplGLResourceTable* table)noexcept :ImplGLResource(table) {}
				};
			}
		}
	}
}
#endif