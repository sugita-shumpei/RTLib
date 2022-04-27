#ifndef TEST_TEST_GLFW_VULKAN_H
#define TEST_TEST_GLFW_VULKAN_H
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <TestGLFW.h>
#include <stdexcept>
#include <iostream>
namespace RTLib {
	namespace Test {

		class TestGLFWVulkanAppExtData: public  RTLib::Test::TestGLFWApplicationExtData {
		public:
			TestGLFWVulkanAppExtData(TestLib::TestApplication* app)noexcept;
			virtual ~TestGLFWVulkanAppExtData()noexcept;

			virtual void Init()         override;
			virtual void Free()noexcept override;
		private:
			struct Impl;
			std::unique_ptr<Impl> m_Impl;
		};

		class TestGLFWVulkanAppInitDelegate :public TestGLFWAppInitDelegate {
		public:
			TestGLFWVulkanAppInitDelegate(TestLib::TestApplication* parent, int width, int height, const char* title)noexcept :TestGLFWAppInitDelegate(parent, width, height, title) {}
			virtual ~TestGLFWVulkanAppInitDelegate()noexcept {}

			virtual void Init() override;
		};
		class TestGLFWVulkanAppMainDelegate :public TestGLFWAppMainDelegate {
		public:
			TestGLFWVulkanAppMainDelegate(TestLib::TestApplication* parent)noexcept :TestGLFWAppMainDelegate(parent) {}
			virtual ~TestGLFWVulkanAppMainDelegate()noexcept {}

			virtual void Main() override;
		};
	}
}
#endif
