#include <RTLib/Ext/GLFW/GLFWContext.h>
#include <GLFW/glfw3.h>
struct RTLib::Ext::GLFW::GLFWContext::Impl
{
	bool isInitialized = false;
};

RTLib::Ext::GLFW::GLFWContext::GLFWContext() noexcept
{
	m_Impl = std::unique_ptr< RTLib::Ext::GLFW::GLFWContext::Impl>(
		new  RTLib::Ext::GLFW::GLFWContext::Impl()
	);
}

auto RTLib::Ext::GLFW::GLFWContext::New() -> GLFWContext*
{
	if (glfwInit() == GLFW_FALSE) {
		return nullptr;
	}
	auto context = new GLFWContext();
	return context;
}

RTLib::Ext::GLFW::GLFWContext::~GLFWContext() noexcept
{
	if (m_Impl->isInitialized) {
		glfwTerminate();
		m_Impl->isInitialized = false;
	}
	m_Impl.reset();
}

bool RTLib::Ext::GLFW::GLFWContext::Initialize()
{
	if (m_Impl->isInitialized) { return true; }
	if (glfwInit() == GLFW_FALSE) { return false; }
	m_Impl->isInitialized = true;
	return true;
}

void RTLib::Ext::GLFW::GLFWContext::Terminate()
{
	if (m_Impl->isInitialized) {
		glfwTerminate();
		m_Impl->isInitialized = false;
	}
}

void RTLib::Ext::GLFW::GLFWContext::Update()
{
	glfwPollEvents();
}
