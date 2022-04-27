#include <TestApplication.h>
#include <TestGLFW.h>

RTLib::Test::TestGLFWApplication::TestGLFWApplication() noexcept
{
	m_IsGlfwInit = false;
	m_Window     = nullptr;
}

RTLib::Test::TestGLFWApplication::~TestGLFWApplication() noexcept
{}

void RTLib::Test::TestGLFWApplication::Init()
{
	if (glfwInit() != GLFW_TRUE) {
		throw std::runtime_error("Failed To Initialize GLFW!");
	}
	m_IsGlfwInit = true;
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_Window = glfwCreateWindow(800, 600, "Title", nullptr, nullptr);
	if (!m_Window) {
		throw std::runtime_error("Failed To Create Window!");
	}
}

void RTLib::Test::TestGLFWApplication::Main()
{
	while (!glfwWindowShouldClose(m_Window)) {
		glfwPollEvents();
	}
}

void RTLib::Test::TestGLFWApplication::Free() noexcept
{
	if (m_Window) {
		glfwDestroyWindow(m_Window);
		m_Window = nullptr;
	}
	if (m_IsGlfwInit) {
		glfwTerminate();
		m_IsGlfwInit = false;
	}
}

void RTLib::Test::TestGLFWAppInitDelegate::Init()
{
	InitGLFW();
	InitWindow(
		{ {GLFW_CLIENT_API,GLFW_NO_API} }
	);
}

auto RTLib::Test::TestGLFWAppInitDelegate::GetWindow() const noexcept -> GLFWwindow*
{
	if (!GetParent()) {
		return nullptr;
	}
	auto app = static_cast<const TestGLFWApplication*>(GetParent());
	return app->m_Window;
}

void RTLib::Test::TestGLFWAppInitDelegate::InitGLFW()
{
	if (!GetParent()) {
		return;
	}
	auto app = static_cast<TestGLFWApplication*>(GetParent());
	if (app->m_IsGlfwInit) {
		return;
	}
	if (glfwInit() != GLFW_TRUE) {
		throw std::runtime_error("Failed To Initialize GLFW!");
	}
	app->m_IsGlfwInit = true;
}

void RTLib::Test::TestGLFWAppInitDelegate::InitWindow(const std::unordered_map<int, int>& windowHints)
{
	if (!GetParent()) {
		return;
	}
	auto app = static_cast<TestGLFWApplication*>(GetParent());
	if (app->m_Window) {
		return;
	}
	for (const auto& [hint, value] : windowHints) {
		glfwWindowHint(hint, value);
	}
	app->m_Window = glfwCreateWindow(m_Width, m_Height, m_Title, nullptr, nullptr);
	if (!app->m_Window) {
		throw std::runtime_error("Failed To Create Window!");
	}
}

void RTLib::Test::TestGLFWAppInitDelegate::MakeContext() 
{
	if (!GetParent()) {
		return;
	}
	auto app = static_cast<TestGLFWApplication*>(GetParent());
	if (!app->m_Window) {
		throw std::runtime_error("Failed To Get Window Handle!");
	}
	glfwMakeContextCurrent(app->m_Window);
}

void RTLib::Test::TestGLFWAppInitDelegate::ShowWindow()
{
	if (!GetParent()) {
		return;
	}
	auto app = static_cast<TestGLFWApplication*>(GetParent());
	if (!app->m_Window) {
		throw std::runtime_error("Failed To Get Window Handle!");
	}
	glfwShowWindow(app->m_Window);
}

void RTLib::Test::TestGLFWAppMainDelegate::Main()
{
	if (!GetParent()) {
		return;
	}
	auto app = static_cast<TestGLFWApplication*>(GetParent());
	while (!glfwWindowShouldClose(app->m_Window)) {
		glfwPollEvents();
	}
}

bool RTLib::Test::TestGLFWAppMainDelegate::ShouldClose()
{
	if (!GetWindow()) {
		return false;
	}
	return glfwWindowShouldClose(GetWindow());
}

void RTLib::Test::TestGLFWAppMainDelegate::SwapBuffers()
{
	if (!GetWindow()) {
		return ;
	}
	return glfwSwapBuffers(GetWindow());
}

void RTLib::Test::TestGLFWAppMainDelegate::PollEvents()
{
	return glfwPollEvents();
}

auto RTLib::Test::TestGLFWAppMainDelegate::GetWindow() const noexcept -> GLFWwindow*
{
	if (!GetParent()) {
		return nullptr;
	}
	auto app = static_cast<const TestGLFWApplication*>(GetParent());
	return app->m_Window;
}

void RTLib::Test::TestGLFWAppFreeDelegate::FreeWindow()noexcept
{
    if (!GetParent()) {
        return;
    }
    auto app = static_cast<TestGLFWApplication*>(GetParent());
    if (app->m_Window) {
        glfwDestroyWindow(app->m_Window);
        app->m_Window = nullptr;
    }
}
void RTLib::Test::TestGLFWAppFreeDelegate::FreeGLFW()noexcept
{
    if (!GetParent()) {
        return;
    }
    auto app = static_cast<TestGLFWApplication*>(GetParent());
    if (app->m_IsGlfwInit) {
        glfwTerminate();
        app->m_IsGlfwInit = false;
    }
}

void RTLib::Test::TestGLFWAppFreeDelegate::Free()noexcept
{
    FreeWindow();
    FreeGLFW();
}

auto RTLib::Test::TestGLFWAppExtensionData::GetWindow() const noexcept -> GLFWwindow*
{
    if (!GetParent()) {
        return nullptr;
    }
    auto app = static_cast<const TestGLFWApplication*>(GetParent());
    return app->m_Window;
}
