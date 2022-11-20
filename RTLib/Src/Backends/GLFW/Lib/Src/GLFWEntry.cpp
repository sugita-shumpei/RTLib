#include <RTLib/Backends/GLFW/GLFWEntry.h>
#include <RTLib/Backends/GLFW/Inputs/GLFWKeyboard.h>
#include <RTLib/Backends/GLFW/Window/GLFWWindow.h>
#include <RTLib/Window/Window.h>
#include "Internals/GLFWInternals.h"
#include <memory>
#include <cassert>
struct RTLib::Backends::Glfw::Entry::Impl {
	Impl() noexcept : globalKeyboard{ new Inputs::Keyboard() }, currentWindow{ nullptr } {
		int res = glfwInit();
		assert(res == GLFW_TRUE);
	}
	~Impl()noexcept {
		globalKeyboard.reset();
		glfwTerminate();
	}
	Window::Window*                   currentWindow;
	std::unique_ptr<Inputs::Keyboard> globalKeyboard;
};
RTLib::Backends::Glfw::Entry::Entry() noexcept:m_Impl{new Impl()}
{
	GetProcAddress = glfwGetProcAddress;
}

RTLib::Backends::Glfw::Entry::~Entry()noexcept
{
	m_Impl.reset();
}

void RTLib::Backends::Glfw::Entry::PollEvents() noexcept
{
	glfwPollEvents();
}

void RTLib::Backends::Glfw::Entry::WaitEvents() noexcept
{
	glfwWaitEvents();
}

auto RTLib::Backends::Glfw::Entry::GetCurrentWindow() const noexcept -> RTLib::Window::Window*
{
	return m_Impl->currentWindow;
}

void RTLib::Backends::Glfw::Entry::SetCurrentWindow(RTLib::Window::Window* window) noexcept
{
	if (window == m_Impl->currentWindow) { return; }
	GLFWwindow* tmpWindow = nullptr;
	if (window) {
		tmpWindow = static_cast<GLFWwindow*>(window->GetHandle());
	}
	glfwMakeContextCurrent(tmpWindow);
	m_Impl->currentWindow = static_cast<RTLib::Backends::Glfw::Window::Window*>(window);
}

auto RTLib::Backends::Glfw::Entry::GetGlobalKeyboard() const noexcept -> RTLib::Inputs::Keyboard*
{
	return m_Impl->globalKeyboard.get();
}

auto RTLib::Backends::Glfw::Entry::GetWindowKeyboard(RTLib::Window::Window* window) const noexcept -> RTLib::Inputs::Keyboard*
{
	if (!window) { return nullptr; }
	return static_cast<RTLib::Backends::Glfw::Window::Window*>(window)->Internal_GetKeyboard();
}

auto RTLib::Backends::Glfw::Entry::CreateWindow(const RTLib::Window::WindowDesc& desc)const->RTLib::Window::Window*
{
	return new RTLib::Backends::Glfw::Window::Window(desc);
}
auto RTLib::Backends::Glfw::Entry::CreateWindowUnique(const RTLib::Window::WindowDesc& desc)const->std::unique_ptr<RTLib::Window::Window>
{
	return std::unique_ptr<RTLib::Window::Window>(CreateWindow(desc));
}
