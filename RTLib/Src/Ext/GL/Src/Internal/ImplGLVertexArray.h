#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_VERTEX_ARRAY_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_VERTEX_ARRAY_H
#include "../Internal/ImplGLBindable.h"
#include "../Internal/ImplGLBuffer.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			namespace Internal
			{
				struct ImplGLVertexBindingInfo
				{
					GLsizei  stride;
				};
				struct ImplGLVertexAttributeFormat
				{
					GLuint    attribIndex;
					GLint     size;
					GLenum    type;
					GLboolean normalized;
					GLuint    relativeOffset;
				};
				class ImplGLVertexArray : public ImplGLBindable
				{
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, const ImplGLBindingPoint* bpBuffer)->ImplGLVertexArray*;
					virtual ~ImplGLVertexArray() noexcept;
					bool Bind();
					bool IsBindable()const noexcept;
					
					bool SetVertexAttribBinding(GLuint attribIndex, GLuint bindIndex);
					bool SetVertexAttribFormat( GLuint attribIndex, GLint size, GLenum type, GLboolean normalized, GLuint relativeOffset = 0);
					bool SetVertexBuffer(GLuint bindIndex, ImplGLBuffer* vertexBuffer, GLsizei stride, GLintptr offset = 0);
					bool SetIndexBuffer(ImplGLBuffer* indexBuffer);
					bool Enable();

					bool IsEnabled()const noexcept { return m_IsEnabled; }

					bool DrawArrays(GLenum mode, GLsizei count, GLint first = 0);
					bool DrawElements(GLenum mode, GLenum  type, GLsizei count, uintptr_t indexOffset = 0);
				protected:
					ImplGLVertexArray(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, const ImplGLBindingPoint* bpBuffer) noexcept;
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
					struct VertexAttribFormatInfo
					{
						GLuint        attribIndex;
						GLint		  size;
						GLenum        type;
						GLboolean     normalized;
						GLuint        relativeOffset;
					};
					struct VertexBindingInfo {
						GLuint        bindIndex;
						ImplGLBuffer* vertexBuffer;
						GLsizei       stride;
						GLintptr      offset;
					};
				private:
					const ImplGLBindingPoint*                          m_BPBuffer             = nullptr;
					std::unordered_map<GLuint, GLuint>                 m_VertexAttribBindings = {};
					std::unordered_map<GLuint, VertexAttribFormatInfo> m_VertexAttributes     = {};
					std::unordered_map<GLuint, VertexBindingInfo>      m_VertexBindings       = {};
					ImplGLBuffer*                                      m_IndexBuffer          = nullptr;
					bool                                               m_IsEnabled            = false;
				};
			}
		}
	}
}
#endif