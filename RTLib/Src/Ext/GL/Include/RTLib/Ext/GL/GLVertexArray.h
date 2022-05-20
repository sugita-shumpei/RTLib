#ifndef RTLIB_EXT_GL_GL_VERTEX_ARRAY_H
#define RTLIB_EXT_GL_GL_VERTEX_ARRAY_H
#include <RTLib/Core/Common.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <unordered_map>
namespace RTLib
{
    namespace Ext
    {
        namespace GL
        {
            class GLContext;
            class GLBuffer;
            class GLVertexArray : public Core::BaseObject
            {
                friend class GLContext;
                RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLVertexArray, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_VERTEX_ARRAY);
            public:
                static auto New(GLContext *context) -> GLVertexArray *;
                virtual ~GLVertexArray()noexcept;
                
                void Destroy() noexcept;
                bool IsBindable() const noexcept;
                bool SetVertexAttribBinding(GLuint attribIndex, GLuint bindIndex);
                bool SetVertexAttribFormat(GLuint attribIndex, GLint size, GLenum type, GLboolean normalized, GLuint relativeOffset = 0);
                bool SetVertexBuffer(GLuint bindIndex, GLBuffer *vertexBuffer, GLsizei stride, GLintptr offset = 0);
                bool SetIndexBuffer(GLBuffer *indexBuffer);
                bool Enable();
                bool IsEnabled() const noexcept;
            protected:
                GLVertexArray(GLContext *context, GLuint resId) noexcept;
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
                        GL_PATCHES};
                    for (auto validMode : validModes)
                    {
                        if (validMode == mode)
                        {
                            return true;
                        }
                    }
                    return false;
                }
                static inline constexpr bool IsValidType(GLenum type)
                {
                    constexpr GLenum validTypes[] = {
                        GL_UNSIGNED_BYTE,
                        GL_UNSIGNED_SHORT,
                        GL_UNSIGNED_INT};
                    for (auto validType : validTypes)
                    {
                        if (validType == type)
                        {
                            return true;
                        }
                    }
                    return false;
                }
                auto GetResId() const noexcept -> GLuint { return m_ResId; }
            private:
                void Bind();
            private:
                
                struct VertexAttribFormatInfo
                {
                    GLuint attribIndex;
                    GLint size;
                    GLenum type;
                    GLboolean normalized;
                    GLuint relativeOffset;
                };
                struct VertexBindingInfo
                {
                    GLuint bindIndex;
                    GLBuffer *vertexBuffer;
                    GLsizei stride;
                    GLintptr offset;
                };
            private:
                GLContext *m_Context = nullptr;
                std::unordered_map<GLuint, GLuint> m_VertexAttribBindings = {};
                std::unordered_map<GLuint, VertexAttribFormatInfo> m_VertexAttributes = {};
                std::unordered_map<GLuint, VertexBindingInfo> m_VertexBindings = {};
                GLBuffer *m_IndexBuffer = nullptr;
                GLuint m_ResId = 0;
                bool m_IsEnabled = false;
                bool m_IsBinded  = false;
            };
        }
    }
}
#endif
