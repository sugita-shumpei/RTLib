#ifndef RTLIB_EXT_GL_GL_NATIVES_H
#define RTLIB_EXT_GL_GL_NATIVES_H
#include <RTLib/Ext/GL/GLCommon.h>
#include <unordered_map>
namespace RTLib
{
    namespace Ext
    {
        namespace GL
        {
            class GLBuffer ;
            class GLImage  ;
            class GLTexture;
            class GLProgram;
            class GLShader ;
            class GLVertexArray ;
            class GLFramebuffer ;
            class GLRenderbuffer;
            struct GLNatives
            {
                static auto GetResId(GLBuffer*  buf)->GLuint;
                static auto GetResId(GLImage*   img)->GLuint;
                static auto GetResId(GLTexture* tex)->GLuint;
                static auto GetResId(GLProgram* prg)->GLuint;
                static auto GetResId(GLShader * shd)->GLuint;
                static auto GetResId(GLVertexArray* vao)->GLuint;
                static auto GetResId(GLFramebuffer* fbo)->GLuint;
                static auto GetResId(GLRenderbuffer* rbo)->GLuint;

            };
        }
    }
}
#endif
