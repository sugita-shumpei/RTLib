#include <RTLib/Ext/GLFW/GL/GLFWOpenGLContext.h>
#include <RTLib/Ext/GLFW/GL/GLFWOpenGLWindow.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
auto RTLib::Ext::GLFW::GL::GLFWOpenGLContext::New(GLFWOpenGLWindow* window) -> GLFWOpenGLContext*
{
	auto context = new GLFWOpenGLContext(window);
	if (!context) { return nullptr; }
	return context;
}

RTLib::Ext::GLFW::GL::GLFWOpenGLContext::~GLFWOpenGLContext() noexcept
{
	
	m_Window = nullptr;
}

RTLib::Ext::GLFW::GL::GLFWOpenGLContext::GLFWOpenGLContext(GLFWOpenGLWindow* window) noexcept:Ext::GL::GLContext()
{
	m_Window = window;
}

bool RTLib::Ext::GLFW::GL::GLFWOpenGLContext::InitLoader()
{
	if (!m_Window) { return false; }
	m_Window->SetCurrent();
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		return false;
	}
	return true;
}

void RTLib::Ext::GLFW::GL::GLFWOpenGLContext::FreeLoader()
{
	if (m_Window) {
		if (RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::GetCurrent() == m_Window)
		{
			glfwMakeContextCurrent(nullptr);
		}
		m_Window = nullptr;
	}
}
