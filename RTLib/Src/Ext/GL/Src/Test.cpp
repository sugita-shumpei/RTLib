#include <TestConfig.h>
#include <RTLib/Core/Common.h>
#include "Internal/ImplGLUtils.h"
#include "Internal/ImplGLBuffer.h"
#include "Internal/ImplGLTexture.h"
#include "Internal/ImplGLFramebuffer.h"
#include "Internal/ImplGLRenderbuffer.h"
#include "Internal/ImplGLSampler.h"
#include "Internal/ImplGLContext.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
auto LoadShaderSource(const char* filename)->std::vector<GLchar>
{
	auto shaderSource = std::vector<GLchar>();
	auto sourceFile   = std::ifstream(filename, std::ios::binary);
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
auto LoadShaderBinary(const char* filename)->std::vector<uint32_t>
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
int main() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    std::vector<std::pair<int, int>> glVersions = {
        {4,6},{4,5},{4,4},{4,3},{4,2},{4,1},{4,0},
        {3,3},{3,2},{3,1},{3,0},
        {2,1},{2,0}
    };
    GLFWwindow* window = nullptr;
    for (auto& [versionMajor,versionMinor]:glVersions){
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, versionMajor);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, versionMinor);
        window = glfwCreateWindow(256, 256, "NONE", NULL, NULL);
        if (window){ break;}
    }
	glfwMakeContextCurrent(window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "ERROR!";
	}
	{
		auto context = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLContext>(RTLib::Ext::GL::Internal::ImplGLContext::New());
		{
			auto testVSSource = std::vector<GLchar>();
			auto testFSSource = std::vector<GLchar>();
			if (context->IsSupportedVersion(4,6)) {
				/*For Windows + Latest GPU*/
				testVSSource = LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test460.vert");
				testFSSource = LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test460.frag");
			}
			else if (context->IsSupportedVersion(4, 2)) {
				testVSSource = LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test420.vert");
				testFSSource = LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test420.frag");
			}
			else if (context->IsSupportedVersion(4, 1)) {
				/*For Mac (M1)*/
				testVSSource = LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test410.vert");
				testFSSource = LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test410.frag");
			}
			else if (context->IsSupportedVersion(3, 3)) {
				testVSSource = LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test330.vert");
				testFSSource = LoadShaderSource(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test330.frag");
			}
			else {
				std::cerr << "Failed To Support Any Shader!\n";
			}

			auto testVSBinary = LoadShaderBinary(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test460.vert.spv");
			auto testFSBinary = LoadShaderBinary(RTLIB_EXT_GL_TEST_CONFIG_SHADER_DIR"/Test460.frag.spv");

			auto vShader = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLShader>(context->CreateShader(GL_VERTEX_SHADER));
			RTLIB_DEBUG_ASSERT_IF_FAILED(vShader!=nullptr);
			vShader->SetName("vShader");
			{
                if(!context->IsSpirvSupported()){
                    RTLIB_DEBUG_ASSERT_IF_FAILED(vShader->ResetBinarySPV(testVSBinary));
                    RTLIB_DEBUG_ASSERT_IF_FAILED(vShader->Specialize("main"));
                }else{
                    RTLIB_DEBUG_ASSERT_IF_FAILED(vShader->ResetSourceGLSL(testVSSource));
                    std::string log;
                    auto res = (vShader->Compile(log));
                    std::cout << log << std::endl;
                    RTLIB_DEBUG_ASSERT_IF_FAILED(res);
                }
			}

			auto fShader = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLShader>(context->CreateShader(GL_FRAGMENT_SHADER));
			RTLIB_DEBUG_ASSERT_IF_FAILED(fShader!= nullptr);
			fShader->SetName("fShader");
			{
				/*TEST: SHADER*/
                if(!context->IsSpirvSupported()){
                    RTLIB_DEBUG_ASSERT_IF_FAILED(fShader->ResetBinarySPV(testFSBinary));
                    RTLIB_DEBUG_ASSERT_IF_FAILED(fShader->Specialize("main"));
                }else{
                    RTLIB_DEBUG_ASSERT_IF_FAILED(fShader->ResetSourceGLSL(testFSSource));
                    std::string log;
                    auto res = (fShader->Compile(log));
                    std::cout << log << std::endl;
                    RTLIB_DEBUG_ASSERT_IF_FAILED(res);
                }
			}

			auto gProgram = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLGraphicsProgram>(context->CreateGraphicsProgram());
			RTLIB_DEBUG_ASSERT_IF_FAILED(gProgram!=nullptr);
			gProgram->SetName("gProgram");
			RTLIB_DEBUG_ASSERT_IF_FAILED(gProgram->AttachShader(vShader.get()));
			RTLIB_DEBUG_ASSERT_IF_FAILED(gProgram->AttachShader(fShader.get()));
			{
				std::string infoLog;
				RTLIB_DEBUG_ASSERT_IF_FAILED(gProgram->Link(infoLog));
				std::cout << infoLog << std::endl;
			}
			RTLIB_DEBUG_ASSERT_IF_FAILED(gProgram->Enable());
			gProgram->Disable();

			auto sgProgram = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLSeparateProgram>(context->CreateSeparateProgram());
			RTLIB_DEBUG_ASSERT_IF_FAILED(sgProgram != nullptr);
			sgProgram->SetName("sgProgram");
			RTLIB_DEBUG_ASSERT_IF_FAILED(sgProgram->AttachShader(vShader.get()));
			RTLIB_DEBUG_ASSERT_IF_FAILED(sgProgram->AttachShader(fShader.get()));
			RTLIB_DEBUG_ASSERT_IF_FAILED(sgProgram->Link());

			auto svProgram = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLSeparateProgram>(context->CreateSeparateProgram());
			RTLIB_DEBUG_ASSERT_IF_FAILED(svProgram != nullptr);
			svProgram->SetName("svProgram");
			RTLIB_DEBUG_ASSERT_IF_FAILED(svProgram->AttachShader(vShader.get()));
			RTLIB_DEBUG_ASSERT_IF_FAILED(svProgram->Link());

			auto sfProgram = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLSeparateProgram>(context->CreateSeparateProgram());
			RTLIB_DEBUG_ASSERT_IF_FAILED(sfProgram != nullptr);
			sfProgram->SetName("sfProgram");
			RTLIB_DEBUG_ASSERT_IF_FAILED(sfProgram->AttachShader(fShader.get()));
			RTLIB_DEBUG_ASSERT_IF_FAILED(sfProgram->Link());

			auto gProgramPipeline = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLGraphicsProgramPipeline>(context->CreateGraphicsProgramPipeline());
			RTLIB_DEBUG_ASSERT_IF_FAILED(gProgramPipeline != nullptr);
			gProgramPipeline->SetName("gProgramPipeline");
			RTLIB_DEBUG_ASSERT_IF_FAILED(gProgramPipeline->Attach(GL_VERTEX_SHADER_BIT|GL_FRAGMENT_SHADER_BIT, sgProgram.get()));

			std::vector<float>    meshVertices = std::vector<float>{
				 -1.0f,-1.0f,0.0f,1.0f, 0.0f,0.0f,
				  1.0f,-1.0f,0.0f,0.0f, 1.0f,0.0f,
				  1.0f, 1.0f,0.0f,0.0f, 0.0f,1.0f,
				 -1.0f, 1.0f,0.0f,1.0f, 1.0f,1.0f
			};
			std::vector<uint32_t> meshIndices  = std::vector<uint32_t>{
				0,1,2,2,3,0
			};
			std::vector<float>    uniformData  = std::vector<float>{ 1.0f,0.0f,1.0f,1.0f };

			auto vMeshBuffer = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLBuffer>(context->CreateBuffer(GL_ARRAY_BUFFER));
			vMeshBuffer->SetName("vMeshBuffer");
			RTLIB_DEBUG_ASSERT_IF_FAILED(vMeshBuffer->Allocate(GL_STATIC_DRAW, sizeof(meshVertices[0]) * std::size(meshVertices), meshVertices.data()));

			auto iMeshBuffer = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLBuffer>(context->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER));
			iMeshBuffer->SetName("iMeshBuffer");
			RTLIB_DEBUG_ASSERT_IF_FAILED(iMeshBuffer->Allocate(GL_STATIC_DRAW, sizeof( meshIndices[0]) * std::size(meshIndices), meshIndices.data()));

			auto meshVertexArray = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLVertexArray>(context->CreateVertexArray());
			meshVertexArray->SetName("meshVertexArray");

			RTLIB_DEBUG_ASSERT_IF_FAILED(meshVertexArray->SetIndexBuffer(iMeshBuffer.get()));
			RTLIB_DEBUG_ASSERT_IF_FAILED(meshVertexArray->SetVertexBuffer(0, vMeshBuffer.get(), sizeof(float)*6));
			RTLIB_DEBUG_ASSERT_IF_FAILED(meshVertexArray->SetVertexAttribBinding(0, 0));
			RTLIB_DEBUG_ASSERT_IF_FAILED(meshVertexArray->SetVertexAttribFormat(0,3, GL_FLOAT, GL_FALSE, 0));
			RTLIB_DEBUG_ASSERT_IF_FAILED(meshVertexArray->SetVertexAttribBinding(1, 0));
			RTLIB_DEBUG_ASSERT_IF_FAILED(meshVertexArray->SetVertexAttribFormat(1,3, GL_FLOAT, GL_FALSE, sizeof(float)*3));
			RTLIB_DEBUG_ASSERT_IF_FAILED(meshVertexArray->Enable());

			auto uniformBuffer = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLBuffer>(context->CreateBuffer(GL_UNIFORM_BUFFER));
			uniformBuffer->SetName("uniformBuffer");
			RTLIB_DEBUG_ASSERT_IF_FAILED(uniformBuffer->Allocate(GL_STATIC_DRAW, sizeof(uniformData[0]) * std::size(uniformData), uniformData.data()));
            
            GLuint bindingIndex = 3;
            if(!context->IsSupportedVersion(4,2)){
                auto tFProgram = gProgramPipeline->GetAttachedProgram(GL_FRAGMENT_SHADER_BIT);
                RTLIB_DEBUG_ASSERT_IF_FAILED(tFProgram!=nullptr);
                auto blockIndex = tFProgram->GetUniformBlockIndex("UBO");
                RTLIB_DEBUG_ASSERT_IF_FAILED(tFProgram->SetUniformBlockBinding(blockIndex, bindingIndex));
            }else{
                bindingIndex = 0;
            }
            
            int fbWidth, fbHeight;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
			glViewport(0, 0, fbWidth, fbHeight);
			while (!glfwWindowShouldClose(window)) {
				glClearColor(0.0f, 0.0f, 0.0f,1.0f);
				glClear(GL_COLOR_BUFFER_BIT);
				RTLIB_DEBUG_ASSERT_IF_FAILED(gProgramPipeline->Bind());
				RTLIB_DEBUG_ASSERT_IF_FAILED(   uniformBuffer->BindBase(bindingIndex));
				RTLIB_DEBUG_ASSERT_IF_FAILED( meshVertexArray->DrawElements(GL_TRIANGLES, GL_UNSIGNED_INT, 6, 0));
				glfwSwapBuffers(window);
				glfwPollEvents();
			}

			context.reset();
		}
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}
