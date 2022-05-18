#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <GLFW/glfw3.h>
#include <unordered_map>
static auto CreateGLFWWindowWithHints(int width, int height, const char* title, const std::unordered_map<int, int>& windowHint)->GLFWwindow* {
	for (auto& [key, value] : windowHint) {
		glfwWindowHint(key, value);
	}
	return glfwCreateWindow(width, height, title,nullptr,nullptr);
}
static auto CreateGLFWWindow(int width, int height, const char* title) -> GLFWwindow* {
	GLFWwindow* window = nullptr;
	auto windowHints = std::unordered_map<int, int>();
	windowHints[GLFW_CLIENT_API] = GLFW_OPENGL_API;
	windowHints[GLFW_OPENGL_PROFILE] = GLFW_OPENGL_CORE_PROFILE;
	windowHints[GLFW_OPENGL_FORWARD_COMPAT] = GLFW_TRUE;
	windowHints[GLFW_VISIBLE] = GLFW_FALSE;
	std::vector<std::pair<int, int>> glVersions = {
		{4,6},{4,5},{4,4},{4,3},{4,2},{4,1},{4,0},
		{3,3},{3,2},{3,1},{3,0},
		{2,1},{2,0}
	};
	for (auto& [version_major, version_minor] : glVersions) {
		windowHints[GLFW_CONTEXT_VERSION_MAJOR] = version_major;
		windowHints[GLFW_CONTEXT_VERSION_MINOR] = version_minor;
		window = CreateGLFWWindowWithHints(width, height, "NONE", windowHints);
		if (window) {
			break;
		}
	}
	return window;
}
class TestGLContext : public RTLib::Ext::GL::GLContext
{
public:
	TestGLContext(GLFWwindow* window):RTLib::Ext::GL::GLContext(), m_Window{window}
	{}
	virtual ~TestGLContext()noexcept {

	}
	// GLContext ÇâÓÇµÇƒåpè≥Ç≥ÇÍÇ‹ÇµÇΩ
	virtual bool InitLoader() override
	{
		glfwMakeContextCurrent(m_Window);
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
			return false;
		}
		return true;
	}
	virtual void FreeLoader() override
	{

	}
private:
	GLFWwindow* m_Window = nullptr;
};
int main(int argc, const char** argv[]) {
	if (!glfwInit()) {
		return -1;
	}
	bool isFailedToCreateWindow    = false;
	bool isFailedToLoadGLADLibrary = false;
	auto context = std::unique_ptr<RTLib::Ext::GL::GLContext >();
	GLFWwindow* window = nullptr;
	int width  = 1024;
	int height = 1024;
	do {
		window  = CreateGLFWWindow(width, height, "title");
		if (!window) {
			isFailedToCreateWindow = true;
			break;
		}
		context = std::unique_ptr<RTLib::Ext::GL::GLContext >(new TestGLContext(window));
		if (!context->Initialize()) {
			break;
		}

		auto vertexBuffer = std::unique_ptr<RTLib::Ext::GL::GLBuffer>(context->CreateBuffer(RTLib::Ext::GL::GLBufferCreateDesc{
			1024,
			RTLib::Ext::GL::GLBufferUsageVertex        |
			RTLib::Ext::GL::GLBufferUsageGenericCopyDst,
			RTLib::Ext::GL::GLMemoryPropertyDefault,
			nullptr
		}));
		auto staginBuffer = std::unique_ptr<RTLib::Ext::GL::GLBuffer>(context->CreateBuffer(RTLib::Ext::GL::GLBufferCreateDesc{
			1024,
			RTLib::Ext::GL::GLBufferUsageGenericCopySrc,
			RTLib::Ext::GL::GLMemoryPropertyHostRead,
			nullptr
		}));
		context->CopyBuffer(staginBuffer.get(), vertexBuffer.get(), { {0,0,1024} });

		vertexBuffer->Destroy();
		staginBuffer->Destroy();

		glfwShowWindow(window);
		while (!glfwWindowShouldClose(window)) {
			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	} while (false);

	context->Terminate();
	context.reset();
	if (!isFailedToCreateWindow) {
		glfwDestroyWindow(window);
		window = nullptr;
	}
	glfwTerminate();

	return 0;
}