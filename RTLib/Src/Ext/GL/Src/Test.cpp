#include "Internal/ImplGLUtils.h"
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
			/*TEST: BUFFER*/
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
				assert(buffer->Allocate(GL_ARRAY_BUFFER, GL_STATIC_DRAW, srcData.size() * sizeof(float)));
				assert(buffer->CopyFromMemory(srcData.data(),srcData.size() * sizeof(float), 0));
				assert(buffer->CopyToMemory(  dstData.data(),dstData.size() * sizeof(float), 0));

				ShowData(dstData);
				assert(buffer->CopyToMemory(dstData.data(),(dstData.size()/2) * sizeof(float), (dstData.size() / 2) * sizeof(float)));
				ShowData(dstData);
				
				auto buffer2 = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLBuffer>(context->CreateBuffer());
				assert(buffer2->Allocate(GL_ARRAY_BUFFER, GL_STATIC_DRAW, srcData.size() * sizeof(float)));
				assert(buffer2->CopyFromBuffer(buffer.get(), srcData.size() * sizeof(float), 0, 0));
				assert(buffer2->CopyToMemory(dstData.data(), dstData.size() * sizeof(float), 0));
				ShowData(dstData);

				{
					srcData[0] = 100;
					srcData[1] = 99;
					srcData[2] = 98;
					void* pMappedData;
					if (buffer2->MapMemory((void**)&pMappedData, GL_WRITE_ONLY)) {
						std::memcpy(pMappedData, srcData.data(), sizeof(float) * srcData.size());
						assert(buffer2->UnmapMemory());
					}
				}
				{
					void* pMappedData;
					if (buffer2->MapMemory((void**)&pMappedData, GL_READ_ONLY)) {
						std::memcpy(dstData.data(), pMappedData, sizeof(float) * dstData.size());
						assert(buffer2->UnmapMemory());
					}
					ShowData(dstData);
				}
			}
			/*TEST: TEXTURE*/
			{
				{
					static_assert(RTLib::Ext::GL::Internal::GetGLFormatTypeInfo(GL_RGBA32F, GL_FLOAT).num_bases == 4);
					auto ShowData  = [](const auto& data) {
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
					texture->SetName("Texture1");
					assert(texture->Allocate(GL_RGBA32F, 3, 1, 4, 4, 1));
					assert(texture->CopyImageFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
					assert(texture->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
					assert(texture->CopyImageFromMemory(srcData2.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));

					auto texture2 = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_2D_ARRAY));
					texture2->SetName("Texture2");
					//EXPLICIT BIND(FOR OPTIMIZATION)
					if (texture2->Bind()) {
						assert(texture2->Allocate(GL_RGBA32F, 3, 4, 4, 4, 1));
						//CPU¨GPU
						assert(texture2->CopyImageFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
						assert(texture2->CopyImageFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 1, 1, 4, 4));
						assert(texture2->CopyImageFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 2, 1, 4, 4));
						assert(texture2->CopyImageFromMemory(srcData0.data(), GL_RGBA, GL_FLOAT, 0, 3, 1, 4, 4));
						assert(texture2->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
						assert(texture2->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 1, 1, 2, 2));
						assert(texture2->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 2, 1, 2, 2));
						assert(texture2->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 1, 3, 1, 2, 2));
						assert(texture2->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));
						assert(texture2->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 2, 1, 1, 1, 1));
						assert(texture2->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 2, 2, 1, 1, 1));
						assert(texture2->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_FLOAT, 2, 3, 1, 1, 1));
						//GPU¨CPU
						assert(texture2->CopyImageToMemory(dstData.data(), GL_RGBA, GL_FLOAT, 0));
						texture2->Unbind();
					}
					else {
						throw std::runtime_error("Failed To Bind Texture2");
					}

					ShowData(dstData);
					ClearData(dstData);

					auto buffer = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLBuffer>(context->CreateBuffer());

					assert(texture2->Bind());
					assert(buffer->Allocate(GL_PIXEL_PACK_BUFFER, GL_STATIC_DRAW, dstData.size() * sizeof(float)));
					assert(texture2->CopyImageToBuffer(buffer.get(), GL_RGBA, GL_FLOAT, 0));
					assert(buffer->CopyToMemory(dstData.data(), dstData.size() * sizeof(float)));
					texture2->Unbind();

					ShowData(dstData);
				}
				{
					auto ShowData = [](const auto& data) {
						for (auto i = 0; i < data.size(); ++i) {
							std::cout << (float)data[i] << " ";
						}
						std::cout << std::endl;
					};
					auto ClearData = [](auto& data) {
						for (auto i = 0; i < data.size(); ++i) {
							data[i] = 0.0f;
						}
					};
					using GLr10g10b10a2 = RTLib::Ext::GL::Internal::GLTypeInfo<GL_UNSIGNED_INT_10_10_10_2>::type;
					std::vector<GLr10g10b10a2> srcData0 = {
						GLr10g10b10a2(4,31,21,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),
						GLr10g10b10a2(4,31,21,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),
						GLr10g10b10a2(4,31,21,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),
						GLr10g10b10a2(4,31,21,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),
					};
					std::vector<GLr10g10b10a2> srcData1 = {
						GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),GLr10g10b10a2(4,3,2,1),
					};
					std::vector<GLr10g10b10a2> srcData2 = {
						GLr10g10b10a2(4,3,2,1)
					};
					static_assert(GLr10g10b10a2(4, 3, 2, 1).GetR() == 4);
					static_assert(GLr10g10b10a2(4, 3, 2, 1).GetG() == 3);
					static_assert(GLr10g10b10a2(4, 3, 2, 1).GetB() == 2);
					static_assert(GLr10g10b10a2(4, 3, 2, 1).GetA() == 1);
					std::vector<unsigned short> dstData = std::vector<unsigned short>(4 * 4 * 4);
					constexpr auto v = RTLib::Ext::GL::Internal::GetGLFormatTypeSize(GL_RGB10_A2, GL_UNSIGNED_INT_10_10_10_2);
					auto texture = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_2D));
					assert(texture->Bind());
					assert(texture->Allocate(GL_RGB10_A2, 3, 1, 4, 4, 1));
					assert(texture->CopyImageFromMemory(srcData0.data(), GL_RGBA, GL_UNSIGNED_INT_10_10_10_2, 0, 0, 1, 4, 4));
					assert(texture->CopyImageFromMemory(srcData1.data(), GL_RGBA, GL_UNSIGNED_INT_10_10_10_2, 1, 0, 1, 2, 2));
					assert(texture->CopyImageFromMemory(srcData2.data(), GL_RGBA, GL_UNSIGNED_INT_10_10_10_2, 2, 0, 1, 1, 1));
					assert(texture->CopyImageToMemory(dstData.data(), GL_RGBA, GL_UNSIGNED_SHORT, 0));
					texture->Unbind();
					ShowData(dstData);
				}
				{
					auto ShowData = [](const auto& data) {
						for (auto i = 0; i < data.size(); ++i) {
							std::cout << data[i].GetDepth() << " ";
						}
						std::cout << std::endl;
					};
					auto ClearData = [](auto& data) {
						for (auto i = 0; i < data.size(); ++i) {
							data[i] = 0.0f;
						}
					};
					using GLD32S8 = RTLib::Ext::GL::Internal::GLTypeInfo<GL_FLOAT_32_UNSIGNED_INT_24_8_REV>::type;
					std::vector<GLD32S8> srcData0 = {
						GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),
						GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),
						GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),
						GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),
						GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),
					};
					std::vector<GLD32S8> srcData1 = {
						GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),GLD32S8(1.0f,3),
					};
					std::vector<GLD32S8> srcData2 = {
						GLD32S8(1.0f,3)
					};
					std::vector<GLD32S8> dstData = std::vector<GLD32S8>(4 * 4);
					auto texture = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_2D));
					assert(texture->Bind());
					assert(texture->Allocate(GL_DEPTH32F_STENCIL8, 3, 1, 4, 4, 1));
					assert(texture->CopyImageFromMemory(srcData0.data(), GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV, 0, 0, 1, 4, 4));
					assert(texture->CopyImageFromMemory(srcData1.data(), GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV, 1, 0, 1, 2, 2));
					assert(texture->CopyImageFromMemory(srcData2.data(), GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV, 2, 0, 1, 1, 1));
					assert(texture->CopyImageToMemory(dstData.data(), GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV, 0));
					texture->Unbind();
					ShowData(dstData);
				}
				{
					static_assert(RTLib::Ext::GL::Internal::GetGLFormatTypeInfo(GL_RGBA32F, GL_FLOAT).num_bases == 4);
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

					auto texture = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLTexture>(context->CreateTexture(GL_TEXTURE_CUBE_MAP));
					texture->SetName("TextureCube1");
					if (texture->Bind()) {
						assert(texture->Allocate(GL_RGBA32F, 3, 1, 4, 4, 1));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_X, srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_X, srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_X, srcData2.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));

						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, srcData2.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));

						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, srcData2.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));

						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, srcData2.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));

						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, srcData2.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));

						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, srcData0.data(), GL_RGBA, GL_FLOAT, 0, 0, 1, 4, 4));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, srcData1.data(), GL_RGBA, GL_FLOAT, 1, 0, 1, 2, 2));
						assert(texture->CopyFaceImageFromMemory(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, srcData2.data(), GL_RGBA, GL_FLOAT, 2, 0, 1, 1, 1));
						texture->Unbind();
					}
					else {
						throw std::runtime_error("Failed To Bind Texture!");
					}

				}
			}
			/*TEST: FRAME BUFFER*/
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
			{
				auto vertexArray = std::unique_ptr<RTLib::Ext::GL::Internal::ImplGLVertexArray>(context->CreateVertexArray());
				vertexArray->Bind();
				vertexArray->Unbind();
			}
		}
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}