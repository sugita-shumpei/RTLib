#include <TestGLFW_Vulkan.h>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

static VkBool32 VKAPI_CALL VKAPI_ATTR DebugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
                                                         VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
                                                         const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
                                                         void*                                            pUserData)
{
    std::cerr << "[" << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity)) << "]";
    std::cerr << "[" << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes)) << "]: ";
    std::cerr << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

struct RTLib::Test::TestGLFWVulkanAppExtendedData::Impl
{
    vk::DynamicLoader                m_DynamicLoaderVK;
    vk::UniqueInstance               m_InstanceVK;
    vk::UniqueDebugUtilsMessengerEXT m_DebugUtilsMessengerEXTVK;
    std::vector<const char*>         m_InstExtPNames;
    std::vector<const char*>         m_InstLyrPNames;
};

RTLib::Test::TestGLFWVulkanAppExtendedData::TestGLFWVulkanAppExtendedData(TestLib::TestApplication* parent)noexcept:Test::TestGLFWAppExtendedData(parent){
    m_Impl = std::unique_ptr<Impl>(new Impl());
}
RTLib::Test::TestGLFWVulkanAppExtendedData::~TestGLFWVulkanAppExtendedData()noexcept{
    m_Impl.reset();
}
void RTLib::Test::TestGLFWVulkanAppExtendedData::InitDynamicLoader()
{
    auto vkGetInstanceProcAddr = m_Impl->m_DynamicLoaderVK.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
}

bool RTLib::Test::TestGLFWVulkanAppInitDelegate::SupportInstExtName(const std::string& extName)const noexcept{
    return std::find_if(std::begin(m_InstExtProps),std::end(m_InstExtProps),[extName](const auto& extProp){
        return std::string(extProp.extensionName) == extName;
    }) != std::end(m_InstExtProps);
}

bool RTLib::Test::TestGLFWVulkanAppInitDelegate::SupportInstLyrName(const std::string& lyrName)const noexcept{
    return std::find_if(std::begin(m_InstLyrProps),std::end(m_InstLyrProps),[lyrName](const auto& lyrProp){
        return std::string(lyrProp.layerName) == lyrName;
    }) != std::end(m_InstLyrProps);
}

void RTLib::Test::TestGLFWVulkanAppInitDelegate::Init()
{
    InitDynamicLoader();
	InitGLFW();
    InitInstance();
	InitWindow({
		{GLFW_CLIENT_API,GLFW_NO_API},
		{GLFW_VISIBLE   ,GLFW_FALSE}
	});
	ShowWindow();
}
void RTLib::Test::TestGLFWVulkanAppInitDelegate::InitDynamicLoader()
{
    if (!GetParent()){
        return ;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtendedData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWVulkanAppExtendedData*>(app->GetExtendedData());
    appExtData->InitDynamicLoader();
    m_InstApiVersion = vk::enumerateInstanceVersion();
    m_InstExtProps   = vk::enumerateInstanceExtensionProperties();
    m_InstLyrProps   = vk::enumerateInstanceLayerProperties();
}
void RTLib::Test::TestGLFWVulkanAppInitDelegate::InitInstance()
{
    if (!GetParent()){
        return ;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtendedData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWVulkanAppExtendedData*>(app->GetExtendedData());
    std::cout << "Instance: Vulkan " << VK_API_VERSION_MAJOR(  m_InstApiVersion) << "."
                                     << VK_API_VERSION_MINOR(  m_InstApiVersion) << "."
                                     << VK_API_VERSION_VARIANT(m_InstApiVersion) << "\n";
    
    appExtData->m_Impl->m_InstExtPNames = {};
    {
        auto glfwInstExtensionCount = static_cast<uint32_t>(0);
        auto glfwInstExtensionPNames= glfwGetRequiredInstanceExtensions(&glfwInstExtensionCount);
        appExtData->m_Impl->m_InstExtPNames = std::vector<const char*>(glfwInstExtensionPNames,glfwInstExtensionPNames+glfwInstExtensionCount);
#ifndef NDEBUG
        if (SupportInstExtName(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)){
            appExtData->m_Impl->m_InstExtPNames.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }else{
            throw std::runtime_error("Failed To Support Extension: VK_EXT_debug_utils!");
        }
#endif
    }
    appExtData->m_Impl->m_InstLyrPNames = {};
#ifndef NDEBUG
    if (SupportInstLyrName("VK_LAYER_KHRONOS_validation")){
        appExtData->m_Impl->m_InstLyrPNames.push_back("VK_LAYER_KHRONOS_validation");
    }else{
        throw std::runtime_error("Failed To Support Layer: VK_LAYER_KHRONOS_validation!");
    }
#endif
    auto applicationInfo= vk::ApplicationInfo()
        .setApiVersion(m_InstApiVersion)
        .setPEngineName("NO ENGINE")
        .setEngineVersion(VK_MAKE_API_VERSION(0, 1, 0, 0))
        .setPApplicationName("TestGLFW_Vulkan")
        .setApplicationVersion(VK_MAKE_API_VERSION(0, 1, 0, 0))
        .setPNext(nullptr);
    
    auto instanceCreateInfo = vk::InstanceCreateInfo()
        .setPApplicationInfo(&applicationInfo)
        .setPEnabledExtensionNames(appExtData->m_Impl->m_InstExtPNames)
        .setPEnabledLayerNames(appExtData->m_Impl->m_InstLyrPNames);
    
#ifndef NDEBUG
    auto debugUtilsMessengerCreateInfo = vk::DebugUtilsMessengerCreateInfoEXT()
        .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral|vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
        .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError|
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning|
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose|
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
        .setPfnUserCallback(DebugUtilsCallback);
    instanceCreateInfo.pNext = &debugUtilsMessengerCreateInfo;
#endif
    
    appExtData->m_Impl->m_InstanceVK = vk::createInstanceUnique(instanceCreateInfo);
    if (!appExtData->m_Impl->m_InstanceVK ){
        throw std::runtime_error("Failed To Create Instance!");
    }
    
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*appExtData->m_Impl->m_InstanceVK);
    
#ifndef NDEBUG
    appExtData->m_Impl->m_DebugUtilsMessengerEXTVK =appExtData->m_Impl->m_InstanceVK->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo);
#endif
}

void RTLib::Test::TestGLFWVulkanAppMainDelegate::Main()
{
	while (!ShouldClose()) {
		SwapBuffers();
		PollEvents();
	}
}

void RTLib::Test::TestGLFWVulkanAppFreeDelegate::Free()noexcept
{
    FreeWindow();
    FreeInstance();
    FreeGLFW();
}

void RTLib::Test::TestGLFWVulkanAppFreeDelegate::FreeInstance()noexcept
{
    if (!GetParent()){
        return ;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtendedData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWVulkanAppExtendedData*>(app->GetExtendedData());
#ifndef NDEBUG
    appExtData->m_Impl->m_DebugUtilsMessengerEXTVK.reset();
#endif
    appExtData->m_Impl->m_InstanceVK.reset();
}
