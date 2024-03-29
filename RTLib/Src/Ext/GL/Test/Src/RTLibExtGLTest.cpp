#define STB_IMAGE_IMPLEMENTATION
#include <RTLib/Core/Exceptions.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLVertexArray.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLShader.h>
#include <RTLib/Ext/GL/GLProgram.h>
#include <RTLib/Ext/GL/GLProgramPipeline.h>
#include <RTLib/Ext/GL/GLRectRenderer.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <RTLibExtGLTestConfig.h>
#include <stb_image.h>
#include <GLFW/glfw3.h>
#include <unordered_map>
#include <fstream>
#include <cassert>
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
	if (!glfwInit()) {
		return -1;
	}
	bool isFailedToCreateWindow    = false;
	bool isFailedToLoadGLADLibrary = false;
	auto context = std::unique_ptr<RTLib::Ext::GL::GLContext >();
	GLFWwindow* window = nullptr;
	int width  = 512;
	int height = 512;
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
		std::vector<float>    vertexData = { -1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,0.0f,1.0f,1.0f };
		std::vector<float>     colorData = {  1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f };
		std::vector<uint32_t> indexData  = {0,1,2};

		auto vertexBuffer = std::unique_ptr<RTLib::Ext::GL::GLBuffer>(context->CreateBuffer(RTLib::Ext::GL::GLBufferCreateDesc{
			sizeof(vertexData[0])*std::size(vertexData),
			RTLib::Ext::GL::GLBufferUsageVertex     |
			RTLib::Ext::GL::GLBufferUsageGenericCopyDst,
			RTLib::Ext::GL::GLMemoryPropertyDefault,
			vertexData.data()
		}));
		vertexBuffer->SetName("Vertex");
		auto  colorBuffer = std::unique_ptr<RTLib::Ext::GL::GLBuffer>(context->CreateBuffer(RTLib::Ext::GL::GLBufferCreateDesc{
			sizeof(colorData[0]) * std::size(colorData),
			RTLib::Ext::GL::GLBufferUsageVertex |
			RTLib::Ext::GL::GLBufferUsageGenericCopyDst,
			RTLib::Ext::GL::GLMemoryPropertyDefault,
			colorData.data()
		}));
		colorBuffer->SetName("Color");
		auto indexBuffer  = std::unique_ptr<RTLib::Ext::GL::GLBuffer>(context->CreateBuffer(RTLib::Ext::GL::GLBufferCreateDesc{
			sizeof(indexData[0])* std::size(indexData),
			RTLib::Ext::GL::GLBufferUsageIndex |
			RTLib::Ext::GL::GLBufferUsageGenericCopyDst,
			RTLib::Ext::GL::GLMemoryPropertyDefault,
			indexData.data()
		}));
		indexBuffer->SetName("Index");
		auto tex = std::unique_ptr<RTLib::Ext::GL::GLTexture>(nullptr);
		auto texDesc = RTLib::Ext::GL::GLTextureCreateDesc();
		{
			int tWid, tHei, tCmp;
			auto pixels = stbi_load(RTLIB_EXT_GL_TEST_CONFIG_DATA_PATH"/Textures/sample.png", &tWid, &tHei, &tCmp, 4);

		    texDesc.image.imageType      = RTLib::Ext::GL::GLImageType::e2D;
			texDesc.image.extent.width   = tWid;
			texDesc.image.extent.height  = tHei;
			texDesc.image.extent.depth   = 0;
			texDesc.image.arrayLayers    = 0;
			texDesc.image.mipLevels      = 1;
			texDesc.image.format         = RTLib::Ext::GL::GLFormat::eRGBA8;
			texDesc.sampler.magFilter    = RTLib::Core::FilterMode::eLinear;
			texDesc.sampler.minFilter    = RTLib::Core::FilterMode::eLinear;

			tex = std::unique_ptr<RTLib::Ext::GL::GLTexture>(context->CreateTexture(texDesc));
			RTLIB_CORE_ASSERT_IF_FAILED(context->CopyMemoryToImage(tex->GetImage(), { {(void*)pixels,{(uint32_t)0,(uint32_t)0,(uint32_t)1},{(uint32_t)0,(uint32_t)0,(uint32_t)0},{(uint32_t)tWid,(uint32_t)tHei,(uint32_t)0}} }));
			stbi_image_free(pixels);

			auto pixelData   = std::vector<unsigned char>(tWid * tHei*4, 255);
			auto pixelBuffer = std::unique_ptr<RTLib::Ext::GL::GLBuffer>(context->CreateBuffer(RTLib::Ext::GL::GLBufferCreateDesc{
				sizeof(pixelData[0]) * std::size(pixelData),
				RTLib::Ext::GL::GLBufferUsageImageCopySrc ,
				RTLib::Ext::GL::GLMemoryPropertyDefault,
				pixelData.data()
			}));
			auto bufferImageCopy = RTLib::Ext::GL::GLBufferImageCopy();
			bufferImageCopy.bufferOffset = 0;
			bufferImageCopy.bufferRowLength   = 0;
			bufferImageCopy.bufferImageHeight = 0;
			bufferImageCopy.imageOffset = {};
			bufferImageCopy.imageExtent.width  = tWid;
			bufferImageCopy.imageExtent.height = tHei;
			bufferImageCopy.imageExtent.depth  = 0;
			bufferImageCopy.imageSubresources.mipLevel = 0;
			bufferImageCopy.imageSubresources.baseArrayLayer = 0;
			bufferImageCopy.imageSubresources.layerCount     = 1;
			//auto res = context->CopyBufferToImage(pixelBuffer.get(), tex->GetImage(), { bufferImageCopy});
			pixelBuffer->Destroy();
		}
		
		auto   vertexShader = std::unique_ptr<RTLib::Ext::GL::GLShader>(context->CreateShader(RTLib::Ext::GL::GLShaderStageVertex));
        if (vertexShader->ResetBinarySPV(LoadBinary(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test460.vert.spv"))){
            vertexShader->Specialize("main");
        }else{
			std::string log;
            vertexShader->ResetSourceGLSL(LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test410.vert"));
			vertexShader->Compile(log);
			std::cout << log << std::endl;
        }

		auto fragmentShader = std::unique_ptr<RTLib::Ext::GL::GLShader>(context->CreateShader(RTLib::Ext::GL::GLShaderStageFragment));
        if (fragmentShader->ResetBinarySPV(LoadBinary( RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test460.frag.spv"))){
            fragmentShader->Specialize("main");
        }else{
			std::string log;
            fragmentShader->ResetSourceGLSL(LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test410.frag"));
			fragmentShader->Compile(log);
			std::cout << log << std::endl;
        }
        
        auto   vao = std::unique_ptr<RTLib::Ext::GL::GLVertexArray>(context->CreateVertexArray());
        vao->SetIndexBuffer(indexBuffer.get());
        vao->SetVertexBuffer(0, vertexBuffer.get(), 0,0);
        vao->SetVertexAttribFormat( 0, RTLib::Ext::GL::GLVertexFormat::eFloat32x3, false);
        vao->SetVertexAttribBinding(0, 0);
        vao->SetVertexBuffer(1,  colorBuffer.get(), 0,0);
        vao->SetVertexAttribFormat( 1, RTLib::Ext::GL::GLVertexFormat::eFloat32x3, false);
        vao->SetVertexAttribBinding(1, 1);
		RTLIB_CORE_ASSERT_IF_FAILED(vao->Enable());
        
		auto graphicsProgram = std::unique_ptr<RTLib::Ext::GL::GLProgram>(context->CreateProgram());
		graphicsProgram->SetSeparetable(false);
		graphicsProgram->AttachShader(vertexShader.get());
		graphicsProgram->AttachShader(fragmentShader.get());
		RTLIB_CORE_ASSERT_IF_FAILED(graphicsProgram->Link());

		auto vertProgram = std::unique_ptr<RTLib::Ext::GL::GLProgram>(context->CreateProgram());
		vertProgram->SetSeparetable(true);
		vertProgram->AttachShader(vertexShader.get());
		RTLIB_CORE_ASSERT_IF_FAILED(vertProgram->Link());

		auto fragProgram = std::unique_ptr<RTLib::Ext::GL::GLProgram>(context->CreateProgram());
		fragProgram->SetSeparetable(true);
		fragProgram->AttachShader(fragmentShader.get());
		RTLIB_CORE_ASSERT_IF_FAILED(fragProgram->Link());

		auto graphicsPipeline = std::unique_ptr<RTLib::Ext::GL::GLProgramPipeline>(context->CreateProgramPipeline());
		graphicsPipeline->SetProgram(RTLib::Ext::GL::GLShaderStageVertex  , vertProgram.get());
		graphicsPipeline->SetProgram(RTLib::Ext::GL::GLShaderStageFragment, fragProgram.get());

		auto smpLoc0 = fragProgram->GetUniformLocation("smp");
		fragProgram->SetUniformImageUnit(smpLoc0, 0);

		auto smpLoc1 = graphicsProgram->GetUniformLocation("smp");

		auto rectRenderer = std::unique_ptr<RTLib::Ext::GL::GLRectRenderer>(context->CreateRectRenderer());

		glfwShowWindow(window);
		while (!glfwWindowShouldClose(window)) {
			context->SetClearBuffer(RTLib::Ext::GL::GLClearBufferFlagsColor);
			context->SetClearColor( 0.0f, 1.0f, 0.0f, 0.0f);
			rectRenderer->DrawTexture(tex.get());
			glfwSwapBuffers(window);
			glfwPollEvents();
		}

		rectRenderer->Destroy();
		tex->Destroy();
		graphicsProgram->Destroy();
		fragmentShader->Destroy();
		vertexShader->Destroy();
		vertexBuffer->Destroy();
		indexBuffer->Destroy();
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
