#include <TestGLFW_GLAD.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <vector>
#include <utility>
#include <string>
#include <cassert>

void RTLib::Test::TestGLFWGLADAppInitDelegate::Init()
{
	InitGLFW();
	InitGLWindow();
	InitGLAD();
	ShowWindow();
}

void RTLib::Test::TestGLFWGLADAppInitDelegate::InitGLWindow() {
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
}

void RTLib::Test::TestGLFWGLADAppInitDelegate::InitGLAD()
{
	MakeContext();
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		throw std::runtime_error("Failed To Initialize GLAD!");
	}
	auto ctx = std::unique_ptr<RTLib::Ext::GL::GLContext>(new RTLib::Ext::GL::GLContext());
	auto bff1= std::unique_ptr<RTLib::Ext::GL::GLBuffer>(ctx->CreateGLBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(float) * 9, nullptr));
	auto bff2= std::unique_ptr<RTLib::Ext::GL::GLBuffer>(ctx->CreateGLBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(float) * 9, nullptr));
	{
		std::vector<float>  iVertices1{ 0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f };
		std::vector<float>  iVertices2{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };
		assert(bff1->CopyImageFromMemory(iVertices1.data(), sizeof(float) * iVertices1.size()));
		assert(bff2->CopyImageFromMemory(iVertices2.data(), sizeof(float) * iVertices2.size()));
		std::vector<float>  oVertices(9);
		assert(bff1->CopyImageToMemory(  oVertices.data(), sizeof(float) * oVertices.size()));
		for (auto i = 0; i < 9; ++i) {
			std::cout << oVertices[i] << ",";
		}
		std::cout << std::endl;
		assert(bff2->CopyImageToMemory(  oVertices.data(), sizeof(float) * oVertices.size()));
		for (auto i = 0; i < 9; ++i) {
			std::cout << oVertices[i] << ",";
		}
		std::cout << std::endl;
		assert(bff1->CopyImageToBuffer(bff2.get(),sizeof(float)*3, sizeof(float) *3));
		assert(bff2->CopyImageToMemory(oVertices.data(), sizeof(float) * oVertices.size()));
		for (auto i = 0; i < 9; ++i) {
			std::cout << oVertices[i] << ",";
		}
		std::cout << std::endl;
	}
	ctx.reset();
	bff1.reset();
	bff2.reset();
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
