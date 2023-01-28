#ifndef RTLIB_EXT_GLFW_GL_GLFW_OPENGL_CONTEXT_H
#define RTLIB_EXT_GLFW_GL_GLFW_OPENGL_CONTEXT_H
#include <RTLib/Ext/GLFW/GLFWContext.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GLFW/GL/UuidDefinitions.h>
namespace RTLib
{
    namespace Ext
    {
        namespace GLFW
        {
            namespace GL
            {
                class GLFWOpenGLWindow;
                class GLFWOpenGLContext :public Ext::GL::GLContext {

                    RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(Ext::GLFW::GL::GLFWOpenGLContext, Ext::GL::GLContext, RTLIB_TYPE_UUID_RTLIB_EXT_GLFW_GL_GLFW_OPENGL_CONTEXT);
                    static auto New(GLFWOpenGLWindow* window)->GLFWOpenGLContext*;
                    virtual ~GLFWOpenGLContext()noexcept;
                    // GLContext を介して継承されました
                    virtual bool InitLoader() override;
                    virtual void FreeLoader() override;
                private:
                    GLFWOpenGLContext(GLFWOpenGLWindow* window)noexcept;
                private:
                    GLFWOpenGLWindow* m_Window;

                };

            }
        }
    }
}
#endif
