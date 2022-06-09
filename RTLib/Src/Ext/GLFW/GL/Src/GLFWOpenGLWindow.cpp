#include <RTLib/Ext/GLFW/GL/GLFWOpenGLWindow.h>
#include <RTLib/Ext/GLFW/GL/GLFWOpenGLContext.h>
#include <GLFW/glfw3.h>
struct RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::Impl 
{
	GLFWContext*                      glfwContext = nullptr;
	std::unique_ptr< GLFWOpenGLContext> glContext = nullptr;
};

auto RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::New(GLFWContext* context, const GLFWOpenGLWindowCreateDesc& desc) -> GLFWOpenGLWindow*
{
	if (!context) {
		return nullptr;
	}
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, desc.versionMajor);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, desc.versionMinor);
	glfwWindowHint(GLFW_OPENGL_PROFILE, desc.isCoreProfile?GLFW_OPENGL_CORE_PROFILE:GLFW_OPENGL_ANY_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, desc.isResizable);
	glfwWindowHint(GLFW_VISIBLE  , desc.isVisible);

	auto handle = glfwCreateWindow(desc.width, desc.height, desc.title, nullptr, nullptr);

	if (!handle) {
		glfwDefaultWindowHints();
		return nullptr;
	}
	auto window = new GLFWOpenGLWindow(handle);
	window->m_Impl->glfwContext = context;
	window->m_Impl->glContext   = std::unique_ptr< GLFWOpenGLContext>( Ext::GLFW::GL::GLFWOpenGLContext::New(window));
	window->m_Impl->glContext->Initialize();
	window->Initialize();
	glfwDefaultWindowHints();
	return window;
}
RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::GLFWOpenGLWindow(GLFWwindow* window) noexcept :GLFWWindow(window), m_Impl{ new Impl() } {}

RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::~GLFWOpenGLWindow() noexcept
{
	m_Impl.reset();
}

void RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::Destroy() noexcept
{
	if (m_Impl->glContext) {
		m_Impl->glContext->Terminate();
		m_Impl->glContext.reset();
	}
	m_Impl->glfwContext = nullptr;
	GLFW::GLFWWindow::Destroy();
}

auto RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::GetOpenGLContext() -> GLFWOpenGLContext*
{
	return m_Impl->glContext.get();
}

auto RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::GetOpenGLContext() const -> const GLFWOpenGLContext*
{
	return m_Impl->glContext.get();
}

void RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::SwapBuffers()
{
	glfwSwapBuffers(GetGLFWwindow());
}

void RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::SetCurrent()
{
	glfwMakeContextCurrent(GetGLFWwindow());
}

auto RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::GetCurrent() -> GLFWOpenGLWindow*
{
	auto handle = glfwGetCurrentContext();
	if (!handle) { return nullptr; }
	auto window = reinterpret_cast<GLFWOpenGLWindow*>(glfwGetWindowUserPointer(handle));
	return window;
}
