#include <TestGLFW_Vulkan.h>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
int main(int argc, const char** argv)
{
	auto app = std::make_unique<RTLib::Test::TestGLFWApplication>();
	app->AddInitDelegate<RTLib::Test::TestGLFWVulkanAppInitDelegate>(256, 256, "title");
	app->AddMainDelegate<RTLib::Test::TestGLFWVulkanAppMainDelegate>();
	app->AddExtData<RTLib::Test::TestGLFWVulkanAppExtData>();
	return app->Run(argc, argv);
}

void RTLib::Test::TestGLFWVulkanAppInitDelegate::Init()
{
	InitGLFW();
	InitWindow({
		{GLFW_CLIENT_API,GLFW_NO_API},
		{GLFW_VISIBLE   ,GLFW_FALSE}
	});
	InitExtData();
	ShowWindow();
}


void RTLib::Test::TestGLFWVulkanAppMainDelegate::Main()
{
	while (!ShouldClose()) {
		SwapBuffers();
		PollEvents();
	}
}

struct RTLib::Test::TestGLFWVulkanAppExtData::Impl {
	vk::DynamicLoader m_DynamicLoader;
};

RTLib::Test::TestGLFWVulkanAppExtData::TestGLFWVulkanAppExtData() noexcept
{
	m_Impl = std::unique_ptr< RTLib::Test::TestGLFWVulkanAppExtData::Impl>(new RTLib::Test::TestGLFWVulkanAppExtData::Impl());
}

RTLib::Test::TestGLFWVulkanAppExtData::~TestGLFWVulkanAppExtData() noexcept
{
	Free();
	m_Impl.reset();
}

void RTLib::Test::TestGLFWVulkanAppExtData::Init()
{
	auto vkGetInstanceProcAddr = m_Impl->m_DynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
}

void RTLib::Test::TestGLFWVulkanAppExtData::Free() noexcept
{
	
}
