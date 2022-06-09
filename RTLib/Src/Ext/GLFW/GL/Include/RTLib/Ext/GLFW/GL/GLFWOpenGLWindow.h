#ifndef RTLIB_EXT_GLFW_GL_GLFW_OPENGL_WINDOW_H
#define RTLIB_EXT_GLFW_GL_GLFW_OPENGL_WINDOW_H
#include <RTLib/Ext/GLFW/GLFWContext.h>
#include <RTLib/Ext/GLFW/GLFWWindow.h>
#include <RTLib/Ext/GLFW/GL/UuidDefinitions.h>
namespace RTLib
{
    namespace Ext
    {
        namespace GLFW
        {
            namespace GL
            {
                class GLFWOpenGLContext;
                struct GLFWOpenGLWindowCreateDesc
                {
                    int          width;
                    int          height;
                    const char * title;
                    int          versionMajor;
                    int          versionMinor;
                    bool         isCoreProfile;
                    bool         isVisible;
                    bool         isResizable;
                };
                class GLFWOpenGLWindow : public RTLib::Ext::GLFW::GLFWWindow
                {
                public:
                    RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(Ext::GLFW::GL::GLFWOpenGLWindow, Ext::GLFW::GLFWWindow, RTLIB_TYPE_UUID_RTLIB_EXT_GLFW_GL_GLFW_OPENGL_WINDOW);
                    static auto New(GLFWContext* context, const GLFWOpenGLWindowCreateDesc& desc)->GLFWOpenGLWindow*;

                    virtual ~GLFWOpenGLWindow() noexcept;
                    virtual void Destroy()noexcept override;

                    void SetCurrent();
                    static auto GetCurrent()->GLFWOpenGLWindow*;

                    auto GetOpenGLContext() -> GLFWOpenGLContext *;
                    auto GetOpenGLContext() const -> const GLFWOpenGLContext *;

                    void SwapBuffers();
                private:
                    GLFWOpenGLWindow(GLFWwindow* window) noexcept;
                private:
                    struct Impl;
                    std::unique_ptr<Impl> m_Impl;
                };
            }
        }
    }
}
#endif
