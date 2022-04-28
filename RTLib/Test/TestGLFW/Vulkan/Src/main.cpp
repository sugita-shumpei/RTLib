#include <TestGLFW_Vulkan.h>
int main(int argc, const char** argv)
{
    auto app = std::make_unique<RTLib::Test::TestGLFWApplication>();
    app->AddInitDelegate<RTLib::Test::TestGLFWVulkanAppInitDelegate>(256, 256, "title");
    app->AddMainDelegate<RTLib::Test::TestGLFWVulkanAppMainDelegate>();
    app->AddFreeDelegate<RTLib::Test::TestGLFWVulkanAppFreeDelegate>();
    app->AddExtendedData<RTLib::Test::TestGLFWVulkanAppExtendedData>();
    {
        auto featuresChain = RTLib::Test::Vulkan::FeaturesChain();
        featuresChain.Set(vk::PhysicalDeviceFeatures2().setFeatures(vk::PhysicalDeviceFeatures().setGeometryShader(VK_TRUE)));
    }
    return app->Run(argc, argv);
}