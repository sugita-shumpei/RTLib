#include <TestGLFW_Vulkan.h>
#include <tuple>
static VkBool32 VKAPI_CALL VKAPI_ATTR DebugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
                                                         VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
                                                         const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
                                                         void*                                            pUserData)
{
    std::cerr << "[" << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity)) << "]";
    std::cerr << "[" << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes))           << "]: ";
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
    vk::PhysicalDevice               m_PhysDeviceVK;
    Vulkan::FeaturesChain            m_DeviFeatsChain;
    std::vector<const char*>         m_DeviExtPNames;
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
    return RTLib::Test::Vulkan::findExtName(m_InstExtProps, extName);
}

bool RTLib::Test::TestGLFWVulkanAppInitDelegate::SupportInstLyrName(const std::string& lyrName)const noexcept{
    return RTLib::Test::Vulkan::findLyrName(m_InstLyrProps, lyrName);
}

void RTLib::Test::TestGLFWVulkanAppInitDelegate::Init()
{
    InitDynamicLoader();
	InitGLFW();
    InitInstance();
	InitWindow({
		{GLFW_CLIENT_API,GLFW_NO_API},
		{GLFW_VISIBLE   ,GLFW_FALSE }
	});
    InitPhysDevice();
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
            throw std::runtime_error("Error: Failed To Support Extension: VK_EXT_debug_utils!");
        }
#endif
    }
    appExtData->m_Impl->m_InstLyrPNames = {};
#ifndef NDEBUG
    if (SupportInstLyrName("VK_LAYER_KHRONOS_validation")){
        appExtData->m_Impl->m_InstLyrPNames.push_back("VK_LAYER_KHRONOS_validation");
    }else{
        throw std::runtime_error("Error: Failed To Support Layer: VK_LAYER_KHRONOS_validation!");
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
        throw std::runtime_error("Error: Failed To Create Instance!");
    }
    
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*appExtData->m_Impl->m_InstanceVK);
    
#ifndef NDEBUG
    appExtData->m_Impl->m_DebugUtilsMessengerEXTVK =appExtData->m_Impl->m_InstanceVK->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo);
    if (!appExtData->m_Impl->m_DebugUtilsMessengerEXTVK) {
        throw std::runtime_error("Error: Failed To Create DebugUtilsMessengerEXT!");
    }
#endif
}

void RTLib::Test::TestGLFWVulkanAppInitDelegate::InitPhysDevice()
{
    if (!GetParent()) {
        return;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtendedData()) {
        return;
    }

    auto appExtData  = static_cast<RTLib::Test::TestGLFWVulkanAppExtendedData*>(app->GetExtendedData());
    if (!appExtData->m_Impl->m_InstanceVK) {
        std::cerr << "Warning: Failed To Get Instance!" << std::endl;
        return;
    }
    auto physDevices = appExtData->m_Impl->m_InstanceVK->enumeratePhysicalDevices();
    if ( physDevices.empty()) {
        throw std::runtime_error("Error: Failed To Get Any Device!");
    }
    appExtData->m_Impl->m_PhysDeviceVK = physDevices[0];

    auto featsSet = appExtData->m_Impl->m_PhysDeviceVK.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR,
        vk::PhysicalDeviceDescriptorIndexingFeaturesEXT,
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        vk::PhysicalDeviceRayQueryFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceDynamicRenderingFeaturesKHR
    >();

    auto physProps    = appExtData->m_Impl->m_PhysDeviceVK.getProperties();
    auto physExtProps = appExtData->m_Impl->m_PhysDeviceVK.enumerateDeviceExtensionProperties();

    appExtData->m_Impl->m_DeviFeatsChain.Set(featsSet.get<vk::PhysicalDeviceFeatures2>());

    if (!RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        throw std::runtime_error("Error: Failed To Support Swapchain!");
    }
    appExtData->m_Impl->m_DeviExtPNames.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    if (physProps.apiVersion      >= VK_API_VERSION_1_2) {
        auto vk11Feats = featsSet.get<vk::PhysicalDeviceVulkan11Features>();
        auto vk12Feats = featsSet.get<vk::PhysicalDeviceVulkan12Features>();
        appExtData->m_Impl->m_DeviFeatsChain.Set(vk11Feats);
        appExtData->m_Impl->m_DeviFeatsChain.Set(vk12Feats);
#ifdef NDEBUG
        appExtData->m_Impl->m_DeviFeatsChain.Map<vk::PhysicalDeviceVulkan12Features>([](auto& feats) {
            feats.bufferDeviceAddressCaptureReplay = VK_FALSE;
        });
#endif
    }
    else if (physProps.apiVersion >= VK_API_VERSION_1_0) {
        if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back( VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
            appExtData->m_Impl->m_DeviFeatsChain.Set(featsSet.get<vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR>());
#ifdef NDEBUG
            appExtData->m_Impl->m_DeviFeatsChain.Map<vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR>([](auto& feats) {
                feats.bufferDeviceAddressCaptureReplay = VK_FALSE;
            });
#endif
        }
        if (RTLib::Test::Vulkan::findExtName(physExtProps,VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
            appExtData->m_Impl->m_DeviFeatsChain.Set(featsSet.get<vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>());
        }
        if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_SPIRV_1_4_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back( VK_KHR_SPIRV_1_4_EXTENSION_NAME);
            if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME)) {
                appExtData->m_Impl->m_DeviExtPNames.push_back( VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
            }
        }
    }
    if (physProps.apiVersion      >= VK_API_VERSION_1_3) {
        auto vk13Feats = featsSet.get<vk::PhysicalDeviceVulkan13Features>();
        appExtData->m_Impl->m_DeviFeatsChain.Set(vk13Feats);
    }
    else if (physProps.apiVersion >= VK_API_VERSION_1_0){
        if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
            auto vkDrFeats = featsSet.get<vk::PhysicalDeviceDynamicRenderingFeaturesKHR>();
            appExtData->m_Impl->m_DeviFeatsChain.Set(vkDrFeats);
        }
    }
    if (physProps.apiVersion      >= VK_API_VERSION_1_1) {
        if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back( VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
        }
        if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back( VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        }
        if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back( VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            auto srcFeats = featsSet.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>();
            appExtData->m_Impl->m_DeviFeatsChain.Set(srcFeats);
#ifdef NDEBUG
            appExtData->m_Impl->m_DeviFeatsChain.Map<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>([&srcFeats](auto& dstFeats) {
                dstFeats.rayTracingPipelineShaderGroupHandleCaptureReplay      = VK_FALSE;
                dstFeats.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
            });
#endif
        }
        if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back( VK_KHR_RAY_QUERY_EXTENSION_NAME);
            appExtData->m_Impl->m_DeviFeatsChain.Set(featsSet.get<vk::PhysicalDeviceRayQueryFeaturesKHR>());
        }
        if (RTLib::Test::Vulkan::findExtName(physExtProps, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
            appExtData->m_Impl->m_DeviExtPNames.push_back( VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            auto srcFeats = featsSet.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
            appExtData->m_Impl->m_DeviFeatsChain.Set(srcFeats);
#ifdef NDEBUG
            appExtData->m_Impl->m_DeviFeatsChain.Map<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>([&srcFeats](auto& dstFeats) {
                dstFeats.accelerationStructureCaptureReplay = VK_FALSE;
            });
#endif
        }
    }
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
