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
	namespace Ext
	{
		namespace GL
		{
			namespace Internal
			{
				class ImplGLVertexArrayBase : public ImplGLBindableBase
				{
				public:
					friend class ImplGLBindable;

				public:
					virtual ~ImplGLVertexArrayBase() noexcept {}

				protected:
					virtual bool Create() noexcept override
					{
						GLuint resId;
						glGenVertexArrays(1, &resId);
						if (resId == 0)
						{
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void Destroy() noexcept
					{
						GLuint resId = GetResId();
						glDeleteVertexArrays(1, &resId);
						SetResId(0);
					}
					virtual void Bind(GLenum target)
					{
						GLuint resId = GetResId();
						if (resId > 0)
						{
							glBindVertexArray(resId);
						}
					}
					virtual void Unbind(GLenum target)
					{
						glBindVertexArray(0);
					}
				};

				class ImplGLVertexArray : public ImplGLBindable
				{
				public:
					static auto New(ImplGLResourceTable *table, ImplGLBindingPoint *bPoint) -> ImplGLVertexArray *
					{
						if (!table || !bPoint)
						{
							return nullptr;
						}
						auto vertexArray = new ImplGLVertexArray(table, bPoint);
						if (vertexArray)
						{
							vertexArray->InitBase<ImplGLVertexArrayBase>();
							auto res = vertexArray->Create();
							if (!res)
							{
								delete vertexArray;
								return nullptr;
							}
						}
						return vertexArray;
					}
					virtual ~ImplGLVertexArray() noexcept {}
					bool Bind() noexcept
					{
						return ImplGLBindable::Bind(GL_VERTEX_ARRAY);
					}
					bool IsBindable()const noexcept {
						return ImplGLBindable::IsBindable(GL_VERTEX_ARRAY);
					}
					bool DrawArrays(GLenum mode, GLsizei count, GLint first = 0) {
						if (!IsValidMode(mode)) {
							return false;
						}
						bool isBindForDraw = false;
						if (!IsBinded()) {
							if (!IsBindable()) {
								return false;
							}
							isBindForDraw = false;
						}
						else {
							isBindForDraw = true;
						}
						if (!isBindForDraw) {
							glBindVertexArray(GetResId());
						}
						glDrawArrays(mode, first, count);
						if (!isBindForDraw) {
							glBindVertexArray(0);
						}
						return true;
					}
					bool DrawElements(GLenum mode, GLenum  type , GLsizei count, uintptr_t indexOffset = 0)
					{
						if (!IsValidMode(mode) || !IsValidType(type) || count == 0 || !m_IndexBuffer) {
							return false;
						}
						bool isBindForDraw = false;
						if (!IsBinded()) {
							if (!IsBindable()) {
								return false;
							}
							isBindForDraw = false;
						}
						else {
							isBindForDraw = true;
						}
						if (!isBindForDraw) {
							glBindVertexArray(GetResId());
						}
						glDrawElements(mode, count, type, reinterpret_cast<void*>(indexOffset));
						if (!isBindForDraw) {
							glBindVertexArray(0);
						}
						return true;
					}
				protected:
					ImplGLVertexArray(ImplGLResourceTable *table, ImplGLBindingPoint *bPoint) noexcept : ImplGLBindable(table, bPoint) {}
					static inline constexpr bool IsValidMode(GLenum mode)
					{
						constexpr GLenum validModes[] = {
							GL_POINTS, 
							GL_LINE_STRIP, 
							GL_LINE_LOOP, 
							GL_LINES, 
							GL_LINE_STRIP_ADJACENCY, 
							GL_LINES_ADJACENCY, 
							GL_TRIANGLE_STRIP, 
							GL_TRIANGLE_FAN, 
							GL_TRIANGLES, 
							GL_TRIANGLE_STRIP_ADJACENCY, 
							GL_TRIANGLES_ADJACENCY, 
							GL_PATCHES
						};
						for (auto validMode: validModes) { if (validMode == mode) { return true; }} 
						return false;
					}
					static inline constexpr bool IsValidType(GLenum type) {
						constexpr GLenum validTypes[] = {
							GL_UNSIGNED_BYTE, 
							GL_UNSIGNED_SHORT,
							GL_UNSIGNED_INT
						};
						for (auto validType : validTypes) { if (validType == type) { return true; } }
						return false;
					}
				private:
					std::vector<ImplGLBindable*> m_VertexBuffers = {};
					ImplGLBindable*              m_IndexBuffer   = nullptr;
				};
			}
		}
	}
}
#endif