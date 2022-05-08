#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_VERTEX_ARRAY_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_VERTEX_ARRAY_H
#include "../Internal/ImplGLBindable.h"
#include "../Internal/ImplGLBuffer.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <iostream>
namespace RTLib
{
	namespace Ext {
		namespace GL
		{
			namespace Internal {
				class ImplGLVertexArrayBase : public ImplGLBindableBase {
				public:
					friend class ImplGLBindable;
				public:
					virtual ~ImplGLVertexArrayBase()noexcept {}
				protected:
					virtual bool      Create()noexcept override {
						GLuint resId;
						glGenVertexArrays(1, &resId);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void     Destroy()noexcept {
						GLuint resId = GetResId();
						glDeleteVertexArrays(1, &resId);
						SetResId(0);
					}
					virtual void   Bind(GLenum target) {
						GLuint resId = GetResId();
						if (resId > 0) {
							glBindVertexArray(resId);
						}
					}
					virtual void Unbind(GLenum target) {
						glBindVertexArray(0);
					}
				};

				class ImplGLVertexArray: public ImplGLBindable{
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)->ImplGLVertexArray* {
						if (!table || !bPoint) {
							return nullptr;
						}
						auto vertexArray = new ImplGLVertexArray(table, bPoint);
						if (vertexArray) {
							vertexArray->InitBase<ImplGLVertexArrayBase>();
							auto res = vertexArray->Create();
							if (!res) {
								delete vertexArray;
								return nullptr;
							}
						}
						return vertexArray;
					}
					virtual ~ImplGLVertexArray()noexcept {}
					bool Bind()noexcept {
						return ImplGLBindable::Bind(GL_VERTEX_ARRAY);
					}
				protected:
					ImplGLVertexArray(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept :ImplGLBindable(table, bPoint) {}
				private:
					std::vector<ImplGLBindable*>                                  m_Bindables ;
				};
			}
		}
	}
}
#endif