#include <RTLib/Backends/GLFW/GLFWEntry.h>
#include <RTLib/Backends/GLFW/Inputs/GLFWKeyboard.h>
#include <RTLib/Backends/GLFW/Window/GLFWWindow.h>
#include <RTLib/Window/Window.h>
#include "Internals/GLFWInternals.h"
#include <memory>
#include <cassert>
#include <chrono>
#include <optional>
struct RTLib::Backends::Glfw::Entry::Impl {
	Impl() noexcept : globalKeyboard{ new Inputs::Keyboard() }, currentWindow{ nullptr }, prvFrameTime{ 0.0f }, prvDeltaTime{ 0.0f }, isResetTime{true} {
		int res = glfwInit();
		assert(res == GLFW_TRUE);
	}
	~Impl()noexcept {
		globalKeyboard.reset();
		glfwTerminate();
	}
	Window::Window*                    currentWindow ;
	std::unique_ptr<Inputs::Keyboard>  globalKeyboard;
	float                              prvFrameTime  ;
	float                              prvDeltaTime  ;
	bool                               isResetTime;
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
	float frameTime = glfwGetTime();
	if (m_Impl->isResetTime) {
		m_Impl->isResetTime = false;
	}
	else {
		m_Impl->prvDeltaTime = frameTime - m_Impl->prvFrameTime;
	}
	m_Impl->prvFrameTime = frameTime;
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
void RTLib::Backends::Glfw::Entry::SetFrameTime(float frameTime)noexcept
{
	m_Impl->prvFrameTime = frameTime;
	m_Impl->prvDeltaTime = 0.0f;
	m_Impl->isResetTime = true;
	glfwSetTime(frameTime);
}

auto RTLib::Backends::Glfw::Entry::GetDeltaTime()const noexcept -> float
{
	return m_Impl->prvDeltaTime;
}
auto RTLib::Backends::Glfw::Entry::GetFrameTime()const noexcept -> float {
	return m_Impl->prvFrameTime;
}