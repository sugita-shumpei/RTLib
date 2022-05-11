#ifndef RTLIB_EXT_GL_RECT_RENDERER_H
#define RTLIB_EXT_GL_RECT_RENDERER_H
#include <RTLib/Ext/GL/GLProgram.h>
namespace RTLib{
    namespace Ext{
        namespace GL{
            class GLRectRenderer
            {
            public:
                 GLRectRenderer()noexcept {}
                ~GLRectRenderer()noexcept {}
                
            private:
                unsigned int m_VAO;
                unsigned int m_VBO;
                unsigned int m_IBO;
                unsigned int m_Program;
                uint32_t     m_Width;
                uint32_t     m_Height;
            };
        }
    }
}
#endif
