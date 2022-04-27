#ifndef TEST_TEST_GLFW_VULKAN_H
#define TEST_TEST_GLFW_VULKAN_H
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <TestGLFW_Common.h>
#include <stdexcept>
#include <string>
#include <iostream>
namespace RTLib {
	namespace Test {

		class TestGLFWVulkanAppInitDelegate :public TestGLFWAppInitDelegate {
		public:
			TestGLFWVulkanAppInitDelegate(TestLib::TestApplication* parent, int width, int height, const char* title)noexcept :TestGLFWAppInitDelegate(parent, width, height, title) {}
			virtual ~TestGLFWVulkanAppInitDelegate()noexcept {}
			virtual void Init() override;
        private:
            void InitDynamicLoader();
            void InitInstance();
            
            bool SupportInstExtName(const std::string& extName)const noexcept;
            bool SupportInstLyrName(const std::string& lyrName)const noexcept;
        private:
            uint32_t                             m_InstApiVersion = 0;
            std::vector<vk::ExtensionProperties> m_InstExtProps   = {};
            std::vector<vk::LayerProperties>     m_InstLyrProps   = {};
		};
    
		class TestGLFWVulkanAppMainDelegate :public TestGLFWAppMainDelegate {
		public:
			TestGLFWVulkanAppMainDelegate(TestLib::TestApplication* parent)noexcept :TestGLFWAppMainDelegate(parent) {}
			virtual ~TestGLFWVulkanAppMainDelegate()noexcept {}

			virtual void Main() override;
		};
    
        class TestGLFWVulkanAppFreeDelegate :public TestGLFWAppFreeDelegate {
        public:
            TestGLFWVulkanAppFreeDelegate(TestLib::TestApplication* parent)noexcept :TestGLFWAppFreeDelegate(parent) {}
            virtual ~TestGLFWVulkanAppFreeDelegate()noexcept {}

            virtual void Free()noexcept override;
        private:
            void FreeInstance()noexcept;
        };
    
        class TestGLFWVulkanAppExtendedData :public TestGLFWAppExtendedData {
        public:
            TestGLFWVulkanAppExtendedData(TestLib::TestApplication* parent)noexcept;
            virtual ~TestGLFWVulkanAppExtendedData()noexcept;
        private:
            void InitDynamicLoader();
        private:
            friend class TestGLFWVulkanAppInitDelegate;
            friend class TestGLFWVulkanAppFreeDelegate;
            friend class TestGLFWVulkanAppMainDelegate;
        private:
            struct Impl;
            std::unique_ptr<Impl> m_Impl;
        };
	}
}
#endif
