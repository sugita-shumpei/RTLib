#include "Internal/ImplGLBuffer.h"
#include "Internal/ImplGLTexture.h"
#include "Internal/ImplGLFramebuffer.h"
#include "Internal/ImplGLRenderbuffer.h"
#include "Internal/ImplGLSampler.h"
#include "Internal/ImplGLContext.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <cassert>
int main() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	auto window = glfwCreateWindow(256, 256, "NONE", NULL, NULL);
	glfwMakeContextCurrent(window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "ERROR!";
	}
	{
		auto context = RTLib::Ext::GL::Internal::ImplGLContext::New();
		{
			{
				std::vector<float> srcData = { 0,1,2,3,4,5,6,7,8 };
				std::vector<float> dstData = { 0,0,0,0,0,0,0,0,0 };
				auto ShowData = [](const auto& data) {
					for (auto i = 0; i < data.size(); ++i) {
						std::cout << data[i] << " ";
					}
					std::cout << std::endl;
				};

				auto buffer = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLBuffer>(context->CreateBuffer());
				buffer->Bind(GL_ARRAY_BUFFER);
				assert(buffer->Allocate(GL_STATIC_DRAW, srcData.size() * sizeof(float)));
				buffer->Unbind();

				assert(buffer->CopyFromMemory(srcData.data(),srcData.size() * sizeof(float), 0));
				assert(buffer->CopyToMemory(  dstData.data(),dstData.size() * sizeof(float), 0));

				ShowData(dstData);
				assert(buffer->CopyToMemory(dstData.data(),(dstData.size()/2) * sizeof(float), (dstData.size() / 2) * sizeof(float)));
				ShowData(dstData);
				
				auto buffer2 = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLBuffer>(context->CreateBuffer());
				buffer2->Bind(GL_ARRAY_BUFFER);
				assert(buffer2->Allocate(GL_STATIC_DRAW, srcData.size() * sizeof(float)));
				buffer2->Unbind();

				assert(buffer2->CopyFromBuffer(buffer.get(), srcData.size() * sizeof(float), 0, 0));
				assert(buffer2->CopyToMemory(dstData.data(), dstData.size() * sizeof(float), 0));
				ShowData(dstData);

			}
			{
				auto ShowData = [](const auto& data) {
					for (auto i = 0; i < data.size(); ++i) {
						std::cout << data[i] << " ";
					}
					std::cout << std::endl;
				};
				auto ClearData = [](auto& data) {
					for (auto i = 0; i < data.size(); ++i) {
						data[i] = 0.0f;
					}
				};
				std::vector<float> srcData0 = {
					1.0f,2.0f,3.0f,4.0f, 1.0f,2.0f,3.0f,4.0f,1.0f,2.0f,3.0f,4.0f, 1.0f,2.0f,3.0f,4.0f,
					1.0f,2.0f,3.0f,4.0f, 1.0f,2.0f,3.0f,4.0f,1.0f,2.0f,3.0f,4.0f, 1.0f,2.0f,3.0f,4.0f,
					1.0f,2.0f,3.0f,4.0f, 1.0f,2.0f,3.0f,4.0f,1.0f,2.0f,3.0f,4.0f, 1.0f,2.0f,3.0f,4.0f,
					1.0f,2.0f,3.0f,4.0f, 1.0f,2.0f,3.0f,4.0f,1.0f,2.0f,3.0f,4.0f, 1.0f,2.0f,3.0f,4.0f
				};
				std::vector<float> srcData1 = {
					1.0f,2.0f,3.0f,4.0f,1.0f,2.0f,3.0f,4.0f,
					1.0f,2.0f,3.0f,4.0f,1.0f,2.0f,3.0f,4.0f
				};
				std::vector<float> srcData2 = { 
					1.0f,2.0f,3.0f,4.0f
				};
				std::vector<float> dstData = std::vector<float>(4 * 4 * 4 * 4);

				auto texture = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_2D));
				assert(texture->Bind());
				assert(texture->Allocate(GL_RGBA32F, 3, 1, 4, 4, 1));
				assert(texture->CopyFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0,0,1,4,4));
				assert(texture->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1,0,1,2,2));
				assert(texture->CopyFromMemory(srcData2.data(), GL_RGBA, GL_FLOAT, 2,0,1,1,1));
				texture->Unbind();

				auto texture2 = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_2D_ARRAY));

				assert(texture2->Bind());
				assert(texture2->Allocate(GL_RGBA32F, 3, 4, 4, 4, 1));
				assert(texture2->CopyFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
				assert(texture2->CopyFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 1, 1, 4, 4));
				assert(texture2->CopyFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 2, 1, 4, 4));
				assert(texture2->CopyFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 3, 1, 4, 4));

				assert(texture2->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
				assert(texture2->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 1, 1, 2, 2));
				assert(texture2->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 2, 1, 2, 2));
				assert(texture2->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 3, 1, 2, 2));

				assert(texture2->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));
				assert(texture2->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 2, 1, 1, 1, 1));
				assert(texture2->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 2, 2, 1, 1, 1));
				assert(texture2->CopyFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 2, 3, 1, 1, 1));

				assert(texture2->CopyToMemory(dstData.data()  , GL_RGBA, GL_FLOAT, 0));
				texture2->Unbind();

				ShowData(dstData);
				ClearData(dstData);

				auto buffer = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLBuffer>(context->CreateBuffer());

				assert(texture2->Bind());
				assert(  buffer->Bind(GL_PIXEL_PACK_BUFFER));
				assert(  buffer->Allocate(GL_STATIC_DRAW    , dstData.size() * sizeof(float)));
				assert(texture2->CopyToBuffer(buffer.get()  , GL_RGBA, GL_FLOAT, 0));
				assert(  buffer->CopyToMemory(dstData.data(), dstData.size()*sizeof(float)));
				  buffer->Unbind();
				texture2->Unbind();

				ShowData(dstData);
			}
			{
				auto colorTexture     = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_2D));
				assert(colorTexture->Bind());
				assert(colorTexture->Allocate(GL_RGBA8, 1, 1, 256, 256, 1));
				colorTexture->Unbind();

				auto depthTexture = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_2D));
				assert(depthTexture->Bind());
				assert(depthTexture->Allocate(GL_DEPTH_COMPONENT16, 1, 1, 256, 256, 1));
				depthTexture->Unbind();

				auto depthStencilTexture = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_2D));
				assert(depthStencilTexture->Bind());
				assert(depthStencilTexture->Allocate(GL_DEPTH24_STENCIL8, 1, 1, 256, 256, 1));
				depthStencilTexture->Unbind();

				auto framebuffer = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLFramebuffer>(context->CreateFramebuffer());
				assert(framebuffer->Bind(GL_FRAMEBUFFER));
				assert(framebuffer->AttachColorTexture(0, colorTexture.get()));
				assert(framebuffer->AttachDepthTexture(   depthTexture.get()));
				assert(framebuffer->IsCompleted());
				framebuffer->Unbind();

				auto framebuffer2 = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLFramebuffer>(context->CreateFramebuffer());
				assert(framebuffer2->Bind(GL_FRAMEBUFFER));
				assert(framebuffer2->AttachColorTexture(0, colorTexture.get()));
				assert(framebuffer2->AttachDepthStencilTexture(depthStencilTexture.get()));
				assert(framebuffer2->IsCompleted());
				framebuffer2->Unbind();
			}
			{
				auto renderbuffer = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLRenderbuffer>(context->CreateRenderbuffer());
				renderbuffer->Bind();
				renderbuffer->Unbind();
			}

		}
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}