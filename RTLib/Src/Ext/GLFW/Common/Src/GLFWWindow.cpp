#include <RTLib/Ext/GLFW/GLFWWindow.h>
#include <GLFW/glfw3.h>
struct RTLib::Ext::GLFW::GLFWWindow::Impl {
	static void Def_WindowSizeCallback(RTLib::Core::Window* window, int width, int height)
	{
		return;
	}
	static void GLFWWindowSizeCallback(GLFWwindow* window, int width, int height)
	{
		if (!window) { return; }
		auto handle = reinterpret_cast<RTLib::Ext::GLFW::GLFWWindow*>(
			glfwGetWindowUserPointer(window)
		);
		handle->m_Impl->windowSizeCallback(handle, width, height);
		auto nativeHandle = static_cast<RTLib::Ext::GLFW::GLFWWindow*>(handle);
		if ((nativeHandle->m_Impl->size.width != width) || (nativeHandle->m_Impl->size.height != height)) {
			int fbWidth, fbHeight;
			glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
			nativeHandle->m_Impl->size.width    = width;
			nativeHandle->m_Impl->size.height   = height;
			nativeHandle->m_Impl->fbSize.width  = fbWidth;
			nativeHandle->m_Impl->fbSize.height = fbHeight;
		}
	}
	GLFWwindow* window = nullptr;
	RTLib::Core::PfnWindowSizeCallback windowSizeCallback = Def_WindowSizeCallback;
	std::string    title   = "";
	Core::Extent2D size    = {};
	Core::Extent2D fbSize  = {};
	bool isResizable = false;
	bool isVisible = false;
	void* pCustomData = nullptr;
};

RTLib::Ext::GLFW::GLFWWindow::~GLFWWindow() noexcept
{
	m_Impl.reset();
}

void RTLib::Ext::GLFW::GLFWWindow::Destroy()noexcept {
	if (m_Impl->window) {
		if (m_Impl->window == glfwGetCurrentContext()) {
			glfwMakeContextCurrent(nullptr);
		}
		glfwDestroyWindow(m_Impl->window);
		m_Impl->window = nullptr;
	}
}

RTLib::Ext::GLFW::GLFWWindow::GLFWWindow(GLFWwindow* glfwNativeHandle)noexcept
{
	m_Impl          = std::unique_ptr< Impl>(new Impl());
	m_Impl->window  = glfwNativeHandle;
}

void RTLib::Ext::GLFW::GLFWWindow::Initialize() {
	if (m_Impl->window) {
		return;
	}
	glfwSetWindowUserPointer(m_Impl->window, this);
	glfwSetWindowSizeCallback(m_Impl->window, Impl::GLFWWindowSizeCallback);
	int wid, hei;
	glfwGetWindowSize(m_Impl->window, &wid, &hei);
	m_Impl->size.width  = wid;	
	m_Impl->size.height = hei;
	glfwGetFramebufferSize(m_Impl->window, &wid, &hei);
	m_Impl->fbSize.width = wid;
	m_Impl->fbSize.height = hei;
	m_Impl->isResizable = static_cast<bool>(glfwGetWindowAttrib(m_Impl->window, GLFW_RESIZABLE));
	m_Impl->isVisible   = static_cast<bool>(glfwGetWindowAttrib(m_Impl->window, GLFW_VISIBLE));
}

auto RTLib::Ext::GLFW::GLFWWindow::GetGLFWwindow() -> GLFWwindow*
{
	return m_Impl->window;
}

bool RTLib::Ext::GLFW::GLFWWindow::Resize(int width, int height) {
	if ((m_Impl->size.width != width) || (m_Impl->size.height != height)) {
		glfwSetWindowSize(m_Impl->window, width, height);
		return true;
	}
	return false;
}
void RTLib::Ext::GLFW::GLFWWindow::Hide()
{
	if (!m_Impl->isVisible) {
		return;
	}
	glfwHideWindow(m_Impl->window);
}
void RTLib::Ext::GLFW::GLFWWindow::Show()
{
	if (!m_Impl->isVisible) {
		SetVisibility(true);
	}
	glfwShowWindow(m_Impl->window);
}
bool RTLib::Ext::GLFW::GLFWWindow::ShouldClose()
{
	return glfwWindowShouldClose(m_Impl->window);
}
void RTLib::Ext::GLFW::GLFWWindow::SetTitle(const char* title) {
	m_Impl->title = title;
	glfwSetWindowTitle(m_Impl->window, title);
}
auto RTLib::Ext::GLFW::GLFWWindow::GetTitle()const noexcept -> std::string {
	return m_Impl->title;
}
auto RTLib::Ext::GLFW::GLFWWindow::GetSize()->Core::Extent2D {
	return m_Impl->size;
}
auto RTLib::Ext::GLFW::GLFWWindow::GetFramebufferSize()->Core::Extent2D {
	return m_Impl->fbSize;
}
void RTLib::Ext::GLFW::GLFWWindow::SetSizeCallback(Core::PfnWindowSizeCallback callback) {
	if (!callback) { return; }
	m_Impl->windowSizeCallback = callback;
}
void RTLib::Ext::GLFW::GLFWWindow::SetUserPointer(void* pCustomData) {
	m_Impl->pCustomData = pCustomData;
}

void RTLib::Ext::GLFW::GLFWWindow::SetResizable(bool resizable)
{
	glfwSetWindowAttrib(m_Impl->window, GLFW_RESIZABLE, resizable);
}

void RTLib::Ext::GLFW::GLFWWindow::SetVisibility(bool visible)
{
	glfwSetWindowAttrib(m_Impl->window, GLFW_VISIBLE, visible);
}
