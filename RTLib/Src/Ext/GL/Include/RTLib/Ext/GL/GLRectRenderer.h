#ifndef RTLIB_EXT_GL_RECT_RENDERER_H
#define RTLIB_EXT_GL_RECT_RENDERER_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
#include <RTLib/Ext/GL/GLContext.h>
namespace RTLib{
    namespace Ext{
        namespace GL{
            class GLTexture;
            class GLRectRenderer : public Core::BaseObject
            {
            public:
                RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLRectRenderer, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_RECT_RENDERER);
                static auto New(GLContext* ctx, const GLVertexArrayCreateDesc& desc)->GLRectRenderer*;
                virtual ~GLRectRenderer()noexcept;

                void Destroy()noexcept;
                void DrawTexture(GLTexture* texture, const std::array<float,16>& transform = 
                    {1.0f,0.0f,0.0f,0.0f,
                    0.0f,1.0f,0.0f,0.0f,
                    0.0f,0.0f,1.0f,0.0f,
                    0.0f,0.0f,0.0f,1.0f}
                );
                void DrawBound2D(const std::array<float,3>& color, const std::array<float, 16>& transform =
                    { 1.0f,0.0f,0.0f,0.0f,
                    0.0f,1.0f,0.0f,0.0f,
                    0.0f,0.0f,1.0f,0.0f,
                    0.0f,0.0f,0.0f,1.0f }
                );
            private:
                GLRectRenderer(GLContext* ctx)noexcept;
            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };
        }
    }
}
#endif
