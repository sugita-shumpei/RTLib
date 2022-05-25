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
                static auto New(GLContext* ctx)->GLRectRenderer*;
                virtual ~GLRectRenderer()noexcept;

                void Destroy()noexcept;
                void DrawTexture(GLTexture* texture);
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
