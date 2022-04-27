#include <TestGLFW_GLAD.h>
#include <vector>
#include <utility>
int main(int argc, const char** argv)
{
	auto app = std::make_unique<RTLib::Test::TestGLFWApplication>();
	app->AddInitDelegate<RTLib::Test::TestGLFWGLADAppInitDelegate>(256, 256, "title");
	app->AddMainDelegate<RTLib::Test::TestGLFWGLADAppMainDelegate>();
	return app->Run(argc,argv);
}

void RTLib::Test::TestGLFWGLADAppInitDelegate::Init()
{
	InitGLFW();
	auto windowHints = std::unordered_map<int, int>();
	windowHints[GLFW_CLIENT_API]			= GLFW_OPENGL_API;
	windowHints[GLFW_OPENGL_PROFILE]		= GLFW_OPENGL_CORE_PROFILE;
	windowHints[GLFW_OPENGL_FORWARD_COMPAT] = GLFW_TRUE;
	windowHints[GLFW_VISIBLE]				= GLFW_FALSE;
	std::vector<std::pair<int, int>> glVersions = {
		{4,6},{4,5},{4,4},{4,3},{4,2},{4,1},{4,0},
		{3,3},{3,2},{3,1},{3,0},
		{2,1},{2,0}
	};
	for (auto& [version_major, version_minor] : glVersions) {
		bool isSuccess = true;
		try {
			windowHints[GLFW_CONTEXT_VERSION_MAJOR] = version_major;
			windowHints[GLFW_CONTEXT_VERSION_MINOR] = version_minor;
			InitWindow(windowHints);
		}
		catch (std::exception& err) {
			isSuccess = false;
		}
		if (isSuccess) {
			break;
		}
	}
	InitGLAD();
	ShowWindow();
}

void RTLib::Test::TestGLFWGLADAppInitDelegate::InitGLAD()
{
	MakeContext();
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		throw std::runtime_error("Failed To Initialize GLAD!");
	}
}

void RTLib::Test::TestGLFWGLADAppMainDelegate::Main()
{
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	while (!ShouldClose()) {
		glClear(GL_COLOR_BUFFER_BIT);
		SwapBuffers();
		PollEvents ();
	}
}
