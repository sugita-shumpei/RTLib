#define STB_IMAGE_IMPLEMENTATION
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLVertexArray.h>
#include <RTLib/Ext/GL/GLRectRenderer.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLShader.h>
#include <RTLib/Ext/GL/GLProgram.h>
#include <RTLib/Ext/GL/GLProgramPipeline.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDATexture.h>
#include <RTLib/Ext/CUDA/CUDAModule.h>
#include <RTLib/Ext/CUDA/CUDAFunction.h>
#include <RTLib/Ext/CUGL/CUGLBuffer.h>
#include <RTLib/Ext/CUGL/CUGLImage.h>
#include <RTLibExtCUGLTestConfig.h>
#include <stb_image.h>
#include <GLFW/glfw3.h>
#include <random>
#include <unordered_map>
#include <fstream>
#include <cassert>
#include <memory>
static auto CreateGLFWWindowWithHints(int width, int height, const char* title, const std::unordered_map<int, int>& windowHint)->GLFWwindow* {
	for (auto& [key, value] : windowHint) {
		glfwWindowHint(key, value);
	}
	return glfwCreateWindow(width, height, title,nullptr,nullptr);
}
auto LoadShaderSource(const char* filename)->std::vector<GLchar>
{
	auto shaderSource = std::vector<GLchar>();
	auto sourceFile = std::ifstream(filename, std::ios::binary);
	if (sourceFile.is_open()) {
		sourceFile.seekg(0, std::ios::end);
		auto size = static_cast<size_t>(sourceFile.tellg());
		shaderSource.resize(size / sizeof(shaderSource[0]));
		sourceFile.seekg(0, std::ios::beg);
		sourceFile.read((char*)shaderSource.data(), size);
		sourceFile.close();
	}
	return shaderSource;
}
auto LoadBinary(const char* filename)->std::vector<uint32_t>
{
	auto shaderBinary = std::vector<uint32_t>();
	auto sourceFile = std::ifstream(filename, std::ios::binary);
	if (sourceFile.is_open()) {
		sourceFile.seekg(0, std::ios::end);
		auto size = static_cast<size_t>(sourceFile.tellg());
		shaderBinary.resize(size / sizeof(shaderBinary[0]));
		sourceFile.seekg(0, std::ios::beg);
		sourceFile.read((char*)shaderBinary.data(), size);
		sourceFile.close();
	}
	return shaderBinary;
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
	// GLContext を介して継承されました
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
int main(int argc, const char* argv[]) {
	using GLContext = RTLib::Ext::  GL::GLContext;
	using GLBuffer  = RTLib::Ext::  GL::GLBuffer;
	using CUContext = RTLib::Ext::CUDA::CUDAContext;
	using CUModule  = RTLib::Ext::CUDA::CUDAModule;
	using CUBuffer  = RTLib::Ext::CUDA::CUDABuffer;
	using CUGLBuffer = RTLib::Ext::CUGL::CUGLBuffer;
	using CUFunction = RTLib::Ext::CUDA::CUDAFunction;
	if (!glfwInit()) {
		return -1;
	}
	bool isFailedToCreateWindow    = false;
	bool isFailedToLoadGLADLibrary = false;
	auto glContext = std::unique_ptr<GLContext>();
	auto cuContext = std::unique_ptr<CUContext>();
	GLFWwindow* window = nullptr;
	int width  = 512;
	int height = 512;
	do {
		window  = CreateGLFWWindow(width, height, "title");
		if (!window) {
			isFailedToCreateWindow = true;
			break;
		}
		glContext = std::unique_ptr<GLContext>(new TestGLContext(window));
		cuContext = std::unique_ptr<CUContext>(new CUContext());
		if (!cuContext->Initialize()) {
			break;
		}
		if (!glContext->Initialize()) {
			break;
		}
		auto cuModule = std::unique_ptr<CUModule>(cuContext->LoadModuleFromFile(RTLIB_EXT_CUGL_TEST_CONFIG_SHADER_DIR"/../cuda/simpleKernel.ptx"));
		auto cuFunction = std::unique_ptr<CUFunction>();
		{
			cuFunction = std::unique_ptr<CUFunction>(cuModule->LoadFunction("randomKernel"));
		}
		auto seedBuffer = std::unique_ptr<CUBuffer>(cuContext->CreateBuffer(RTLib::Ext::CUDA::CUDABufferCreateDesc{ RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, width * height * sizeof(uint32_t)}));
		{
			auto seedData = std::vector<uint32_t>(width * height);
			auto mt19937 = std::mt19937();
			std::generate(std::begin(seedData), std::end(seedData), mt19937);
			auto seedCopy = RTLib::Ext::CUDA::CUDAMemoryBufferCopy();
			seedCopy.srcData = seedData.data();
			seedCopy.size = width * height * 4;
			seedCopy.dstOffset = 0;
			cuContext->CopyMemoryToBuffer(seedBuffer.get(), { seedCopy });
		}
		auto frameDesc = RTLib::Ext::GL::GLBufferCreateDesc{};
		frameDesc.size = width * height * 4;
		frameDesc.pData = nullptr;
		frameDesc.access = RTLib::Ext::GL::GLMemoryPropertyDefault;
		frameDesc.usage = RTLib::Ext::GL::GLBufferUsageImageCopySrc;

		auto frameBufferGL   = std::unique_ptr<GLBuffer>(glContext->CreateBuffer(frameDesc));
		auto frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(cuContext.get(),frameBufferGL.get(), RTLib::Ext::CUGL::CUGLGraphicsRegisterFlagsNone));

		auto cuStream = std::unique_ptr<RTLib::Ext::CUDA::CUDAStream>(cuContext->CreateStream());

		auto tex = std::unique_ptr<RTLib::Ext::GL::GLTexture>(nullptr);
		{
			auto texDesc = RTLib::Ext::GL::GLTextureCreateDesc();
		    texDesc.image.imageType      = RTLib::Ext::GL::GLImageType::e2D;
			texDesc.image.extent.width   = width;
			texDesc.image.extent.height  = height;
			texDesc.image.extent.depth   = 0;
			texDesc.image.arrayLayers    = 0;
			texDesc.image.mipLevels      = 1;
			texDesc.image.format         = RTLib::Ext::GL::GLFormat::eRGBA8;
			texDesc.sampler.magFilter    = RTLib::Core::FilterMode::eLinear;
			texDesc.sampler.minFilter    = RTLib::Core::FilterMode::eLinear;

			tex = std::unique_ptr<RTLib::Ext::GL::GLTexture>(glContext->CreateTexture(texDesc));
		}
		auto rectRenderer = std::unique_ptr<RTLib::Ext::GL::GLRectRenderer>(glContext->CreateRectRenderer());

		RTLib::Ext::CUDA::CUDAKernelLaunchDesc lncDesc = {};
		{
			CUdeviceptr seedptr = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(seedBuffer.get());
			lncDesc.gridDimX = width;
			lncDesc.gridDimY = height;
			lncDesc.gridDimZ = 1;
			lncDesc.blockDimX = 16;
			lncDesc.blockDimY = 16;
			lncDesc.blockDimZ = 1;
			lncDesc.kernelParams.resize(4);
			lncDesc.kernelParams[0] = &seedptr;
			lncDesc.kernelParams[1] = nullptr;
			lncDesc.kernelParams[2] = &width;
			lncDesc.kernelParams[3] = &height;
			lncDesc.sharedMemBytes = 0;
			lncDesc.stream = cuStream.get();
		}

		glfwShowWindow(window);
		while (!glfwWindowShouldClose(window)) {
			{
				auto frameBufferCU = frameBufferCUGL->Map(cuStream.get());
				{
					CUdeviceptr outputptr = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(frameBufferCU);
					lncDesc.kernelParams[1] = &outputptr;
					cuFunction->Launch(lncDesc);
					cuStream->Synchronize();
				}
				frameBufferCUGL->Unmap(cuStream.get());
			}
			glContext->SetClearBuffer(RTLib::Ext::GL::GLClearBufferFlagsColor);
			glContext->SetClearColor(0.0f, 1.0f, 0.0f, 0.0f);
			
			{
				auto bufferImageCopy = RTLib::Ext::GL::GLBufferImageCopy();

				bufferImageCopy.bufferOffset = 0;
				bufferImageCopy.imageExtent.width = width;
				bufferImageCopy.imageExtent.height = height;
				bufferImageCopy.imageExtent.depth = 0;
				bufferImageCopy.imageSubresources.baseArrayLayer = 0;
				bufferImageCopy.imageSubresources.layerCount = 1;
				bufferImageCopy.imageSubresources.mipLevel = 0;

				glContext->CopyBufferToImage(frameBufferGL.get(), tex->GetImage(), { bufferImageCopy });
			}

			glContext->SetClearBuffer(RTLib::Ext::GL::GLClearBufferFlagsColor);
			glContext->SetClearColor (0.0f, 1.0f, 0.0f, 0.0f);
			rectRenderer->DrawTexture(tex.get());
			glfwSwapBuffers(window);
			glfwPollEvents();
		}
		cuFunction->Destory();
		cuModule->Destory();
		cuStream->Destroy();
		frameBufferCUGL->Destroy();
		frameBufferGL->Destroy();
		rectRenderer->Destroy();
		tex->Destroy();
	} while (false);
	cuContext->Terminate();
	glContext->Terminate();
	glContext.reset();
	cuContext.reset();
	if (!isFailedToCreateWindow) {
		glfwDestroyWindow(window);
		window = nullptr;
	}
	glfwTerminate();
	return 0;
}
