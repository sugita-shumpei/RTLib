#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYEXR_IMPLEMENTATION
#include <RTLibExtOPX7TestApplication.h>

void RTLibExtOPX7TestApplication::CursorPosCallback(RTLib::Core::Window* window, double x, double y)
{
    auto pWindowState = reinterpret_cast<rtlib::test::WindowState*>(window->GetUserPointer());
    if ((pWindowState->curCurPos.x == 0.0f &&
        pWindowState->curCurPos.y == 0.0f))
    {
        pWindowState->curCurPos.x = x;
        pWindowState->curCurPos.y = y;
    }
    else
    {
        pWindowState->delCurPos.x = pWindowState->curCurPos.x - x;
        pWindowState->delCurPos.y = pWindowState->curCurPos.y - y;
        pWindowState->curCurPos.x = x;
        pWindowState->curCurPos.y = y;
    }
}

void RTLibExtOPX7TestApplication::LoadScene(int argc, const char** argv)
{
    m_SceneData = rtlib::test::LoadScene(m_ScenePath);
    if (!m_SceneData.config.enableVis) {
        m_EnableVis = false;
    }
    if (m_SceneData.config.defTracer != "NONE")
    {
        m_CurTracerName = m_SceneData.config.defTracer;
    }
    if (!m_SceneData.config.custom.GetBoolOr("SdTree.Enable",false)) {
        m_EnableTree = false;
    }
    if (!m_SceneData.config.custom.GetBoolOr("MortonTree.Enable", false)) {
        m_EnableGrid = false;
    }
    if (argc > 1) {
        for (int i = 0; i < argc-1; ++i) {
            if (std::string(argv[i]) == "--EnableGrid") {
                m_EnableGrid = false;
                if ((std::string(argv[i + 1]) =="true") || (std::string(argv[i + 1]) == "True")   ||
                    (std::string(argv[i + 1]) ==  "on") || (std::string(argv[i + 1]) ==   "On")   ||
                    (std::string(argv[i + 1]) == "1")) {
                    std::cout << "SUC: --EnableGrid Args is ON\n";
                    m_EnableGrid = true;
                }
                else if ((std::string(argv[i + 1]) == "false") || (std::string(argv[i + 1]) == "False") ||
                    (std::string(argv[i + 1]) == "off") || (std::string(argv[i + 1]) == "Off") ||
                    (std::string(argv[i + 1]) == "0")) {
                    std::cout << "SUC: --EnableGrid Args is OFF\n";
                    m_EnableGrid = false;
                }
                else {
                    std::cout << "BUG: --EnableGrid Args is Missing: Use Default(false)\n";
                }
            }
            if (std::string(argv[i]) == "--EnableTree") {
                m_EnableTree = false;
                if ((std::string(argv[i + 1]) == "true") || (std::string(argv[i + 1]) == "True") ||
                    (std::string(argv[i + 1]) == "on") || (std::string(argv[i + 1]) == "On") ||
                    (std::string(argv[i + 1]) == "1")) {
                    std::cout << "SUC: --EnableTree Args is ON\n";
                    m_EnableTree = true;
                }
                else if ((std::string(argv[i + 1]) == "false") || (std::string(argv[i + 1]) == "False") ||
                    (std::string(argv[i + 1]) == "off") || (std::string(argv[i + 1]) == "Off") ||
                    (std::string(argv[i + 1]) == "0")) {
                    std::cout << "SUC: --EnableTree Args is OFF\n";
                    m_EnableTree = false;
                }
                else {
                    std::cout << "BUG: --EnableTree Args is Missing: Use Default(false)\n";
                }
            }
        }
    }
    auto neeThresholdMeshSize = m_SceneData.config.custom.GetUInt32Or("Nee.ThresholdMeshSize", 0);
    for (auto& [name, asset] : m_SceneData.objAssetManager.GetAssets())
    {
        for (auto& [uniqueName, uniqueRes] : asset.meshGroup->GetUniqueResources())
        {
            if (uniqueRes->variables.GetBoolOr("hasLight", false))
            {
                uniqueRes->variables.SetBool("useNEE", uniqueRes->triIndBuffer.size() > neeThresholdMeshSize);
            }
        }
    }
}

 void RTLibExtOPX7TestApplication::SaveScene()
{
    rtlib::test::SaveScene(m_ScenePath, m_SceneData);
}

 void RTLibExtOPX7TestApplication::InitGLFW()
{
    m_GlfwContext = std::unique_ptr<RTLib::Ext::GLFW::GLFWContext>(RTLib::Ext::GLFW::GLFWContext::New());
    m_GlfwContext->Initialize();
}

 void RTLibExtOPX7TestApplication::FreeGLFW()
{
    if (!m_GlfwContext)
    {
        return;
    }
    m_GlfwContext->Terminate();
    m_GlfwContext.reset();
}

 void RTLibExtOPX7TestApplication::InitOGL4()
{
    m_GlfwWindow = std::unique_ptr<RTLib::Ext::GLFW::GL::GLFWOpenGLWindow>(rtlib::test::CreateGLFWWindow(m_GlfwContext.get(), m_SceneData.config.width, m_SceneData.config.height, "title"));
}

 void RTLibExtOPX7TestApplication::FreeOGL4()
{
    if (!m_GlfwWindow)
    {
        return;
    }
    m_GlfwWindow->Destroy();
    m_GlfwWindow.reset();
}

 void RTLibExtOPX7TestApplication::InitOPX7()
{
    m_Opx7Context = std::unique_ptr<RTLib::Ext::OPX7::OPX7Context>();
    {
        auto contextCreateDesc = RTLib::Ext::OPX7::OPX7ContextCreateDesc();
        {
            contextCreateDesc.validationMode = RTLib::Ext::OPX7::OPX7ContextValidationMode::eALL;
            contextCreateDesc.level = RTLib::Ext::OPX7::OPX7ContextValidationLogLevel::ePrint;
        }
        m_Opx7Context = std::make_unique<RTLib::Ext::OPX7::OPX7Context>(contextCreateDesc);
    }
    m_Opx7Context->Initialize();
}

 void RTLibExtOPX7TestApplication::FreeOPX7()
{
    if (!m_Opx7Context)
    {
        return;
    }
    m_Opx7Context->Terminate();
    m_Opx7Context.reset();
}

 void RTLibExtOPX7TestApplication::InitWorld()
{
    m_ShaderTableLayout = rtlib::test::GetShaderTableLayout(m_SceneData.world, RAY_TYPE_COUNT);
    auto accelBuildOptions = OptixAccelBuildOptions();
    {
        accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelBuildOptions.motionOptions = {};
        accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    }
    m_SceneData.InitExtData(m_Opx7Context.get());
    m_GeometryASMap = m_SceneData.BuildGeometryASs(m_Opx7Context.get(), accelBuildOptions);
    m_InstanceASMap = m_SceneData.BuildInstanceASs(m_Opx7Context.get(), accelBuildOptions, m_ShaderTableLayout.get(), m_GeometryASMap);
    m_TextureMap    = m_SceneData.LoadTextureMap(m_Opx7Context.get());
    auto aabb = rtlib::test::AABB();
    for (auto& instancePath : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instancePath);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto  geometryAS   = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {
            auto tmpAabb = rtlib::test::AABB(m_GeometryASMap.at(geometryAS->GetName()).aabbMin, m_GeometryASMap.at(geometryAS->GetName()).aabbMax).Transform(rtlib::test::GetInstanceTransform(m_SceneData.world, instancePath));
            aabb.Update(tmpAabb.min);
            aabb.Update(tmpAabb.max);
        }
    }
    m_WorldAabbMin = aabb.min;
    m_WorldAabbMax = aabb.max;
}

 void RTLibExtOPX7TestApplication::FreeWorld()
{
    {
        for (auto& [objName, objAsset] : m_SceneData.objAssetManager.GetAssets())
        {
            for (auto& [uniqueName, uniqueRes] : objAsset.meshGroup->GetUniqueResources())
            {
                if (uniqueRes->extData) {
                    uniqueRes->GetExtData<rtlib::test::OPX7MeshUniqueResourceExtData>()->Destroy();
                    uniqueRes->extData.reset();
                }
            }
        }
    }
    {
        for (auto& [name, sphereRes] : m_SceneData.sphereResources)
        {
            if (sphereRes->extData) {
                sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->Destroy();
                sphereRes->extData.reset();
            }
        }
    }
    {
        for (auto& [name, texture] : m_TextureMap)
        {
            texture.Destroy();
        }
        m_TextureMap.clear();
    }
    {
        for (auto& [name, geometryAS] : m_GeometryASMap)
        {
            geometryAS.buffer->Destroy();
        }
        m_GeometryASMap.clear();
    }
    {
        for (auto& [name, instanceAS] : m_InstanceASMap)
        {
            instanceAS.instanceBuffer->Destroy();
            instanceAS.buffer->Destroy();
        }
        m_InstanceASMap.clear();
    }
}

 void RTLibExtOPX7TestApplication::InitLight()
{
    m_lightBuffer = rtlib::test::UploadBuffer<MeshLight>();

    {
        for (auto& instancePath : m_ShaderTableLayout->GetInstanceNames())
        {
            auto& instanceDesc = m_ShaderTableLayout->GetDesc(instancePath);
            auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
            auto geometryAS = instanceData->GetDwGeometryAS();
            if (geometryAS)
            {
                for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                {
                    if (m_SceneData.world.geometryObjModels.count(geometryName) > 0) {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes)
                        {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshName);
                            auto desc = m_ShaderTableLayout->GetDesc(instancePath + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            auto meshLight = MeshLight();
                            bool hasLight = mesh->GetUniqueResource()->variables.GetBoolOr("hasLight", false);
                            bool useNEE   = mesh->GetUniqueResource()->variables.GetBoolOr("useNEE"  , false);
                            if (hasLight && useNEE)
                            {
                                auto meshLight     = MeshLight();
                                meshLight.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                meshLight.normals  = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                meshLight.texCrds  = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                meshLight.indices  = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                meshLight.indCount = mesh->GetUniqueResource()->triIndBuffer.size();
                                meshLight.emission = meshData.materials.front().GetFloat3As<float3>("emitCol");
                                auto emitTexStr    = meshData.materials.front().GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                    ;
                                }
                                meshLight.emissionTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                meshLight.transform = rtlib::test::GetInstanceTransform(m_SceneData.world, instancePath);
                                m_lightBuffer.cpuHandle.push_back(meshLight);
                            }
                        }
                    }
                }
            }
        }
        m_lightBuffer.Upload(m_Opx7Context.get());
    }
}

 void RTLibExtOPX7TestApplication::FreeLight()
{
    if (!m_lightBuffer.gpuHandle)
    {
        return;
    }
    m_lightBuffer.gpuHandle->Destroy();
    m_lightBuffer.gpuHandle.reset();
}

 void RTLibExtOPX7TestApplication::InitGrids()
{
    m_HashBufferCUDA.aabbMin = make_float3(m_WorldAabbMin[0] - 0.005f, m_WorldAabbMin[1] - 0.005f, m_WorldAabbMin[2] - 0.005f);
    m_HashBufferCUDA.aabbMax = make_float3(m_WorldAabbMax[0] + 0.005f, m_WorldAabbMax[1] + 0.005f, m_WorldAabbMax[2] + 0.005f);
    auto hashGridCellSize = static_cast<unsigned int>(128*128*4);
    auto hashGridGridSize = make_uint3(128,64,128);
    if (m_SceneData.config.custom.HasFloat1("HashGrid.CellSize")&&
        m_SceneData.config.custom.HasFloat3("HashGrid.GridSize")) {
        hashGridCellSize = m_SceneData.config.custom.GetFloat1("HashGrid.CellSize");
        hashGridGridSize = make_uint3(
            m_SceneData.config.custom.GetFloat3("HashGrid.GridSize")[0],
            m_SceneData.config.custom.GetFloat3("HashGrid.GridSize")[1],
            m_SceneData.config.custom.GetFloat3("HashGrid.GridSize")[2]
        );
    }
    auto ratioForBudget    = m_SceneData.config.custom.GetFloat1Or("MortonTree.RatioForBudget"   ,0.3f);
    auto iterationForBuilt = m_SceneData.config.custom.GetUInt32Or("MortonTree.IterationForBuilt",3);
    auto fraction          = m_SceneData.config.custom.GetFloat1Or("MortonTree.Fraction"         ,0.3f);
    m_SceneData.config.custom.SetUInt32("MortonTree.MaxLevel", rtlib::test::RTMortonQuadTreeWrapper::kMaxLevel);
    std::cout << "HashGrid.GridSize: (" << hashGridGridSize.x << "," << hashGridGridSize.y << "," << hashGridGridSize.z << ")\n";
    std::cout << "HashGrid.CellSize:  " << hashGridCellSize   << "\n";
    std::cout << "MortonTree.RatioForBudget   :  " << ratioForBudget    << "\n";
    std::cout << "MortonTree.IterationForBuilt:  " << iterationForBuilt << "\n";
    std::cout << "MortonTree.Fraction         :  " << fraction          << "\n";

    m_HashBufferCUDA.Alloc(hashGridGridSize, hashGridCellSize);
    m_HashBufferCUDA.Upload(m_Opx7Context.get());
    {
        auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
        desc.sizeInBytes = sizeof(float4) * m_HashBufferCUDA.checkSumCpuHandle.size();
        m_DiffuseBufferCUDA = std::unique_ptr <RTLib::Ext::CUDA::CUDABuffer>(
            m_Opx7Context->CreateBuffer(desc)
        );
    }

    if (m_EnableGrid) {
        m_MortonQuadTree           = std::make_unique<rtlib::test::RTMortonQuadTreeWrapper>(m_Opx7Context.get(),m_HashBufferCUDA.checkSumCpuHandle.size(), 30, fraction);
        m_MortonQuadTree->Allocate();
        m_MortonQuadTreeController = std::make_unique<rtlib::test::RTMortonQuadTreeController>(
            m_MortonQuadTree.get(), (uint32_t)m_SceneData.config.maxSamples, iterationForBuilt,0, ratioForBudget, m_SceneData.config.samples
        );
    }
}

 void RTLibExtOPX7TestApplication::FreeGrids()
{
    m_HashBufferCUDA.checkSumGpuHandles[0]->Destroy();
    m_HashBufferCUDA.checkSumGpuHandles[1]->Destroy();
    m_HashBufferCUDA.checkSumGpuHandles[0].reset();
    m_HashBufferCUDA.checkSumGpuHandles[1].reset();
    if (m_EnableGrid) {
        m_MortonQuadTree->Destroy();
        m_MortonQuadTree.reset();
        m_MortonQuadTreeController->Destroy();
        m_MortonQuadTreeController.reset();
    }
}

 void RTLibExtOPX7TestApplication::InitSdTree()
 {
     auto fraction          = m_SceneData.config.custom.GetFloat1Or("SdTree.Fraction"         , 0.3f);
     auto ratioForBudget    = m_SceneData.config.custom.GetFloat1Or("SdTree.RatioForBudget"   , 0.3f);
     auto iterationForBuilt = m_SceneData.config.custom.GetUInt32Or("SdTree.IterationForBuilt", 0);

     m_SdTree = std::make_unique<rtlib::test::RTSTreeWrapper>(m_Opx7Context.get(),
         make_float3(m_WorldAabbMin[0] - 0.005f, m_WorldAabbMin[1] - 0.005f, m_WorldAabbMin[2] - 0.005f),
         make_float3(m_WorldAabbMax[0] + 0.005f, m_WorldAabbMax[1] + 0.005f, m_WorldAabbMax[2] + 0.005f),
         20, fraction
     );
     m_SdTree->Upload();
     m_SdTreeController = std::make_unique<rtlib::test::RTSTreeController>(
         m_SdTree.get(), (uint32_t)m_SceneData.config.maxSamples, iterationForBuilt, ratioForBudget,m_SceneData.config.samples
     );
 }

 void RTLibExtOPX7TestApplication::FreeSdTree()
 {
     if (m_SdTree) {
         m_SdTree->Destroy();
         m_SdTree.reset();
         m_SdTreeController.reset();
     }
 }

void RTLibExtOPX7TestApplication::InitPtxString()
{
    m_PtxStringMap["SimpleTrace.ptx"]  = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleTrace.optixir");
    m_PtxStringMap["SimpleGuide.ptx"]  = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleGuide.optixir");
    m_PtxStringMap["SimpleGuide2.ptx"] = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleGuide2.optixir");
}

void RTLibExtOPX7TestApplication::InitDefTracer()
{
    m_TracerMap["DEF"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
    m_TracerMap["DEF"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
    m_TracerMap["DEF"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
    m_TracerMap["DEF"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
    m_TracerMap["DEF"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
    m_TracerMap["DEF"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle| RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    m_TracerMap["DEF"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_TracerMap["DEF"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_TracerMap["DEF"].pipelines["Trace"].linkOptions.maxTraceDepth = 1;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_NONE;
        if (m_EnableGrid) {
            flags |= PARAM_FLAG_USE_GRID;
        }
#ifdef NDEBUG
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        moduleOptions.boundValueEntries.push_back({});
        moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
        moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
        moduleOptions.boundValueEntries.front().sizeInBytes   = sizeof(flags);
        m_TracerMap["DEF"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.DEF", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        m_TracerMap["DEF"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.DEF", moduleOptions);
        m_TracerMap["DEF"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(),   "BuiltIn.Sphere.DEF", moduleOptions);
    }
    m_TracerMap["DEF"].pipelines["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.DEF", "__raygen__default");
    m_TracerMap["DEF"].pipelines["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.DEF", "__exception__ep");
    m_TracerMap["DEF"].pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.DEF", "__miss__radiance");
    m_TracerMap["DEF"].pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.DEF", "__miss__occluded");
    m_TracerMap["DEF"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.DEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.DEF", "");
    m_TracerMap["DEF"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.DEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.DEF", "");
    m_TracerMap["DEF"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere"  , "SimpleKernel.DEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.DEF", "");
    m_TracerMap["DEF"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere"  , "SimpleKernel.DEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.DEF", "");
    m_TracerMap["DEF"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["DEF"].pipelines["Trace"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" 
    };

    m_TracerMap["DEF"].pipelines["Trace"].SetHostRayGenRecordTypeData(   rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_TracerMap["DEF"].pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
    m_TracerMap["DEF"].pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    m_TracerMap["DEF"].pipelines["Trace"].SetHostExceptionRecordTypeData(unsigned int());

    for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {
            for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
            {
                if (m_SceneData.world.geometryObjModels.count(geometryName)>0)
                {
                    auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                    auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                    for (auto& [meshName, meshData] : geometry.meshes)
                    {
                        auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                        auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                        auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                        auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = meshData.materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["DEF"].pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride]+".Triangle", hitgroupData);
                        }
                    }
                }
                if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                    auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                    auto  desc      = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            auto material         = sphereRes->materials[i / desc.recordStride];
                            hitgroupData.type     = rtlib::test::SpecifyMaterialType(material);
                            hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                            hitgroupData.diffuse  = material.GetFloat3As<float3>("diffCol");
                            hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                            hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                            hitgroupData.shinness = material.GetFloat1("shinness");
                            hitgroupData.refIndex = material.GetFloat1("refrIndx");
                            auto diffTexStr       = material.GetString("diffTex");
                            if (diffTexStr == "")
                            {
                                diffTexStr = "White";
                            }
                            auto specTexStr       = material.GetString("specTex");
                            if (specTexStr == "")
                            {
                                specTexStr = "White";
                            }
                            auto emitTexStr       = material.GetString("emitTex");
                            if (emitTexStr == "")
                            {
                                emitTexStr = "White";
                            }
                            hitgroupData.diffuseTex  = m_TextureMap.at(diffTexStr).GetCUtexObject();
                            hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                            hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                        }
                        m_TracerMap["DEF"].pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                    }
                }
            }
        }
    }
    m_TracerMap["DEF"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["DEF"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["DEF"].InitParams(m_Opx7Context.get(),params);
}

void RTLibExtOPX7TestApplication::InitNeeTracer()
{
    m_TracerMap["NEE"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
    m_TracerMap["NEE"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
    m_TracerMap["NEE"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
    m_TracerMap["NEE"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
    m_TracerMap["NEE"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
    m_TracerMap["NEE"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    m_TracerMap["NEE"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_TracerMap["NEE"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_TracerMap["NEE"].pipelines["Trace"].linkOptions.maxTraceDepth = 2;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_NEE;
        if (m_EnableGrid) {
            flags |= PARAM_FLAG_USE_GRID;
        }
#ifdef NDEBUG
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        moduleOptions.boundValueEntries.push_back({});
        moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
        moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
        moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
        m_TracerMap["NEE"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.NEE", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        m_TracerMap["NEE"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.NEE", moduleOptions);
        m_TracerMap["NEE"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.NEE", moduleOptions);
    }
    m_TracerMap["NEE"].pipelines["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.NEE", "__raygen__default");
    m_TracerMap["NEE"].pipelines["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.NEE", "__exception__ep");
    m_TracerMap["NEE"].pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.NEE", "__miss__radiance");
    m_TracerMap["NEE"].pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.NEE", "__miss__occluded");
    m_TracerMap["NEE"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.NEE", "__closesthit__radiance", "", "", "BuiltIn.Triangle.NEE", "");
    m_TracerMap["NEE"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.NEE", "__closesthit__occluded", "", "", "BuiltIn.Triangle.NEE", "");
    m_TracerMap["NEE"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.NEE", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.NEE", "");
    m_TracerMap["NEE"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.NEE", "__closesthit__occluded", "", "", "BuiltIn.Sphere.NEE", "");
    m_TracerMap["NEE"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["NEE"].pipelines["Trace"].shaderTable = this->NewShaderTable();
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" 
    };
    m_TracerMap["NEE"].pipelines["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_TracerMap["NEE"].pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
    m_TracerMap["NEE"].pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    m_TracerMap["NEE"].pipelines["Trace"].SetHostExceptionRecordTypeData( unsigned int());

    for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {
            for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
            {
                if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                {
                    auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                    auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                    for (auto& [meshName, meshData] : geometry.meshes)
                    {
                        auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                        auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                        auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                        auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = meshData.materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["NEE"].pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                        }
                    }
                }
                if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                    auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                    auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            auto material = sphereRes->materials[i / desc.recordStride];
                            hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                            hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                            hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                            hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                            hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                            hitgroupData.shinness = material.GetFloat1("shinness");
                            hitgroupData.refIndex = material.GetFloat1("refrIndx");
                            auto diffTexStr = material.GetString("diffTex");
                            if (diffTexStr == "")
                            {
                                diffTexStr = "White";
                            }
                            auto specTexStr = material.GetString("specTex");
                            if (specTexStr == "")
                            {
                                specTexStr = "White";
                            }
                            auto emitTexStr = material.GetString("emitTex");
                            if (emitTexStr == "")
                            {
                                emitTexStr = "White";
                            }
                            hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                            hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                            hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                        }
                        m_TracerMap["NEE"].pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                    }
                }
            }
        }
    }
    m_TracerMap["NEE"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["NEE"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["NEE"].InitParams(m_Opx7Context.get(), params);
}

void RTLibExtOPX7TestApplication::InitRisTracer()
{
    m_TracerMap["RIS"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
    m_TracerMap["RIS"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
    m_TracerMap["RIS"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
    m_TracerMap["RIS"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
    m_TracerMap["RIS"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
    m_TracerMap["RIS"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    m_TracerMap["RIS"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_TracerMap["RIS"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_TracerMap["RIS"].pipelines["Trace"].linkOptions.maxTraceDepth = 2;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_RIS|PARAM_FLAG_NEE;
        if (m_EnableGrid) {
            flags |= PARAM_FLAG_USE_GRID;
        }
#ifdef NDEBUG
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        moduleOptions.boundValueEntries.push_back({});
        moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
        moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
        moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
        m_TracerMap["RIS"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.RIS", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        m_TracerMap["RIS"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.RIS", moduleOptions);
        m_TracerMap["RIS"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.RIS", moduleOptions);
    }
    m_TracerMap["RIS"].pipelines["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.RIS", "__raygen__default");
    m_TracerMap["RIS"].pipelines["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.RIS", "__exception__ep");
    m_TracerMap["RIS"].pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.RIS", "__miss__radiance");
    m_TracerMap["RIS"].pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.RIS", "__miss__occluded");
    m_TracerMap["RIS"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.RIS", "__closesthit__radiance", "", "", "BuiltIn.Triangle.RIS", "");
    m_TracerMap["RIS"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.RIS", "__closesthit__occluded", "", "", "BuiltIn.Triangle.RIS", "");
    m_TracerMap["RIS"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.RIS", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.RIS", "");
    m_TracerMap["RIS"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.RIS", "__closesthit__occluded", "", "", "BuiltIn.Sphere.RIS", "");
    m_TracerMap["RIS"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["RIS"].pipelines["Trace"].shaderTable = this->NewShaderTable();
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" };
    m_TracerMap["RIS"].pipelines["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_TracerMap["RIS"].pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
    m_TracerMap["RIS"].pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    m_TracerMap["RIS"].pipelines["Trace"].SetHostExceptionRecordTypeData(unsigned int());

    for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {
            for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
            {
                if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                {
                    auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                    auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                    for (auto& [meshName, meshData] : geometry.meshes)
                    {
                        auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                        auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                        auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                        auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = meshData.materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["RIS"].pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                        }
                    }
                }
                if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                    auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                    auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            auto material = sphereRes->materials[i / desc.recordStride];
                            hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                            hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                            hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                            hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                            hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                            hitgroupData.shinness = material.GetFloat1("shinness");
                            hitgroupData.refIndex = material.GetFloat1("refrIndx");
                            auto diffTexStr = material.GetString("diffTex");
                            if (diffTexStr == "")
                            {
                                diffTexStr = "White";
                            }
                            auto specTexStr = material.GetString("specTex");
                            if (specTexStr == "")
                            {
                                specTexStr = "White";
                            }
                            auto emitTexStr = material.GetString("emitTex");
                            if (emitTexStr == "")
                            {
                                emitTexStr = "White";
                            }
                            hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                            hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                            hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                        }
                        m_TracerMap["RIS"].pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                    }
                }
            }
        }
    }
    m_TracerMap["RIS"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["RIS"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["RIS"].InitParams(m_Opx7Context.get(), params);
}

void RTLibExtOPX7TestApplication::InitDbgTracer() {
    m_TracerMap["DBG"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
    m_TracerMap["DBG"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
    m_TracerMap["DBG"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
    m_TracerMap["DBG"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
    m_TracerMap["DBG"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
    m_TracerMap["DBG"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    m_TracerMap["DBG"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_TracerMap["DBG"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_TracerMap["DBG"].pipelines["Trace"].linkOptions.maxTraceDepth = 1;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_NEE;
        if (m_EnableGrid) {
            flags |= PARAM_FLAG_USE_GRID;
        }
#ifdef NDEBUG
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eModerate;
#endif
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        m_TracerMap["DBG"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.DBG", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        m_TracerMap["DBG"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.DBG", moduleOptions);
        m_TracerMap["DBG"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.DBG", moduleOptions);
    }
    m_TracerMap["DBG"].pipelines["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.DBG", "__raygen__debug");
    m_TracerMap["DBG"].pipelines["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.DBG", "__exception__ep");
    m_TracerMap["DBG"].pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Debug", "SimpleKernel.DBG", "__miss__debug");
    m_TracerMap["DBG"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Debug.Triangle", "SimpleKernel.DBG", "__closesthit__debug", "", "", "BuiltIn.Triangle.DBG", "");
    m_TracerMap["DBG"].pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Debug.Sphere", "SimpleKernel.DBG", "__closesthit__debug_sphere", "", "", "BuiltIn.Sphere.DBG", "");
    m_TracerMap["DBG"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["DBG"].pipelines["Trace"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Debug",
        "SimpleKernel.Debug" };

    m_TracerMap["DBG"].pipelines["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_TracerMap["DBG"].pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Debug", MissData{});
    m_TracerMap["DBG"].pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Debug", MissData{});
    m_TracerMap["DBG"].pipelines["Trace"].SetHostExceptionRecordTypeData(unsigned int());
    for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {

            for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
            {

                if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                {
                    auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                    auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                    for (auto& [meshName, meshData] : geometry.meshes)
                    {
                        auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                        auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                        auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                        auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = meshData.materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["DBG"].pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                        }
                    }
                }
                if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                    auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                    auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            auto material = sphereRes->materials[i / desc.recordStride];
                            hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                            hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                            hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                            hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                            hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                            hitgroupData.shinness = material.GetFloat1("shinness");
                            hitgroupData.refIndex = material.GetFloat1("refrIndx");
                            auto diffTexStr = material.GetString("diffTex");
                            if (diffTexStr == "")
                            {
                                diffTexStr = "White";
                            }
                            auto specTexStr = material.GetString("specTex");
                            if (specTexStr == "")
                            {
                                specTexStr = "White";
                            }
                            auto emitTexStr = material.GetString("emitTex");
                            if (emitTexStr == "")
                            {
                                emitTexStr = "White";
                            }
                            hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                            hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                            hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                        }
                        m_TracerMap["DBG"].pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                    }
                }
            }
        }
    }
    m_TracerMap["DBG"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["DBG"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {

        params.flags = PARAM_FLAG_NEE;
    }
    m_TracerMap["DBG"].InitParams(m_Opx7Context.get(), params);
}

void RTLibExtOPX7TestApplication::InitSdTreeDefTracer()
{
    {
        //Build
        {
            m_TracerMap["PGDEF"].pipelines["Build"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGDEF"].pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGDEF"].pipelines["Build"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGDEF"].pipelines["Build"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGDEF"].pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGDEF"].pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGDEF"].pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGDEF"].pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGDEF"].pipelines["Build"].linkOptions.maxTraceDepth = 1;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGDEF"].pipelines["Build"].LoadModule(m_Opx7Context.get(),                      "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGDEF"].pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGDEF"].pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get()  ,   "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            m_TracerMap["PGDEF"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGDEF"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGDEF"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGDEF"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGDEF"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGDEF"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGDEF"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGDEF"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGDEF"].pipelines["Trace"].linkOptions.maxTraceDepth = 1;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE|PARAM_FLAG_BUILD;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGDEF"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGDEF"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGDEF"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            m_TracerMap["PGDEF"].pipelines["Final"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGDEF"].pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGDEF"].pipelines["Final"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGDEF"].pipelines["Final"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGDEF"].pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGDEF"].pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGDEF"].pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGDEF"].pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGDEF"].pipelines["Final"].linkOptions.maxTraceDepth = 1;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_FINAL;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGDEF"].pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGDEF"].pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGDEF"].pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Trace","Build","Final" };
    for (auto& stName : stNames) {
        m_TracerMap["PGDEF"].pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        m_TracerMap["PGDEF"].pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        m_TracerMap["PGDEF"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        m_TracerMap["PGDEF"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        m_TracerMap["PGDEF"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["PGDEF"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["PGDEF"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        m_TracerMap["PGDEF"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    m_TracerMap["PGDEF"].pipelines["Build"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGDEF"].pipelines["Build"].shaderTable = this->NewShaderTable();

    m_TracerMap["PGDEF"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGDEF"].pipelines["Trace"].shaderTable = this->NewShaderTable();
    
    m_TracerMap["PGDEF"].pipelines["Final"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGDEF"].pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    for (auto& stName : stNames) {
        m_TracerMap["PGDEF"].pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        m_TracerMap["PGDEF"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
        m_TracerMap["PGDEF"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        m_TracerMap["PGDEF"].pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
        {
            auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
            auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
            auto geometryAS = instanceData->GetDwGeometryAS();
            if (geometryAS)
            {
                for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                {
                    if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                    {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes)
                        {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                            auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupData = HitgroupData();
                                if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                                {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                    hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                    hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                    hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                    hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                    hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                    hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                    hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                    hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                    hitgroupData.shinness = material.GetFloat1("shinness");
                                    hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                    auto diffTexStr = material.GetString("diffTex");
                                    if (diffTexStr == "")
                                    {
                                        diffTexStr = "White";
                                    }
                                    auto specTexStr = material.GetString("specTex");
                                    if (specTexStr == "")
                                    {
                                        specTexStr = "White";
                                    }
                                    auto emitTexStr = material.GetString("emitTex");
                                    if (emitTexStr == "")
                                    {
                                        emitTexStr = "White";
                                    }
                                    hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                    hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                    hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                                }
                                m_TracerMap["PGDEF"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                            }
                        }
                    }
                    if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                        auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                        auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = sphereRes->materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["PGDEF"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                        }
                    }
                }
            }
        }
    }

    m_TracerMap["PGDEF"].pipelines["Build"].shaderTable->Upload();
    m_TracerMap["PGDEF"].pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["PGDEF"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["PGDEF"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["PGDEF"].pipelines["Final"].shaderTable->Upload();
    m_TracerMap["PGDEF"].pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["PGDEF"].InitParams(m_Opx7Context.get(),params);
}

void RTLibExtOPX7TestApplication::InitSdTreeNeeTracer()
{
    {
        //Build
        {
            m_TracerMap["PGNEE"].pipelines["Build"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGNEE"].pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGNEE"].pipelines["Build"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGNEE"].pipelines["Build"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGNEE"].pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGNEE"].pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGNEE"].pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGNEE"].pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGNEE"].pipelines["Build"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE| PARAM_FLAG_NEE;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGNEE"].pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGNEE"].pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGNEE"].pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            m_TracerMap["PGNEE"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGNEE"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGNEE"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGNEE"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGNEE"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGNEE"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGNEE"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGNEE"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGNEE"].pipelines["Trace"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_NEE;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGNEE"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGNEE"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGNEE"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            m_TracerMap["PGNEE"].pipelines["Final"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGNEE"].pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGNEE"].pipelines["Final"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGNEE"].pipelines["Final"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGNEE"].pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGNEE"].pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGNEE"].pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGNEE"].pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGNEE"].pipelines["Final"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_FINAL | PARAM_FLAG_NEE;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGNEE"].pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGNEE"].pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGNEE"].pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Trace","Build","Final" };
    for (auto& stName : stNames) {
        m_TracerMap["PGNEE"].pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        m_TracerMap["PGNEE"].pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        m_TracerMap["PGNEE"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        m_TracerMap["PGNEE"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        m_TracerMap["PGNEE"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["PGNEE"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["PGNEE"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        m_TracerMap["PGNEE"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    m_TracerMap["PGNEE"].pipelines["Build"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGNEE"].pipelines["Build"].shaderTable = this->NewShaderTable();

    m_TracerMap["PGNEE"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGNEE"].pipelines["Trace"].shaderTable = this->NewShaderTable();

    m_TracerMap["PGNEE"].pipelines["Final"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGNEE"].pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    for (auto& stName : stNames) {
        m_TracerMap["PGNEE"].pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        m_TracerMap["PGNEE"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
        m_TracerMap["PGNEE"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        m_TracerMap["PGNEE"].pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
        {
            auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
            auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
            auto geometryAS = instanceData->GetDwGeometryAS();
            if (geometryAS)
            {
                for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                {
                    if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                    {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes)
                        {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                            auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupData = HitgroupData();
                                if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                                {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                    hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                    hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                    hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                    hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                    hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                    hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                    hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                    hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                    hitgroupData.shinness = material.GetFloat1("shinness");
                                    hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                    auto diffTexStr = material.GetString("diffTex");
                                    if (diffTexStr == "")
                                    {
                                        diffTexStr = "White";
                                    }
                                    auto specTexStr = material.GetString("specTex");
                                    if (specTexStr == "")
                                    {
                                        specTexStr = "White";
                                    }
                                    auto emitTexStr = material.GetString("emitTex");
                                    if (emitTexStr == "")
                                    {
                                        emitTexStr = "White";
                                    }
                                    hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                    hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                    hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                                }
                                m_TracerMap["PGNEE"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                            }
                        }
                    }
                    if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                        auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                        auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = sphereRes->materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["PGNEE"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                        }
                    }
                }
            }
        }
    }

    m_TracerMap["PGNEE"].pipelines["Build"].shaderTable->Upload();
    m_TracerMap["PGNEE"].pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["PGNEE"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["PGNEE"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["PGNEE"].pipelines["Final"].shaderTable->Upload();
    m_TracerMap["PGNEE"].pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["PGNEE"].InitParams(m_Opx7Context.get(), params);
}

void RTLibExtOPX7TestApplication::InitSdTreeRisTracer()
{
    {
        //Build
        {
            m_TracerMap["PGRIS"].pipelines["Build"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGRIS"].pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGRIS"].pipelines["Build"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGRIS"].pipelines["Build"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGRIS"].pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGRIS"].pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGRIS"].pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGRIS"].pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGRIS"].pipelines["Build"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_NEE | PARAM_FLAG_RIS;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGRIS"].pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGRIS"].pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGRIS"].pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            m_TracerMap["PGRIS"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGRIS"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGRIS"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGRIS"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGRIS"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGRIS"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGRIS"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGRIS"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGRIS"].pipelines["Trace"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_NEE | PARAM_FLAG_RIS;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGRIS"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGRIS"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGRIS"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            m_TracerMap["PGRIS"].pipelines["Final"].compileOptions.usesMotionBlur = false;
            m_TracerMap["PGRIS"].pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["PGRIS"].pipelines["Final"].compileOptions.numAttributeValues = 3;
            m_TracerMap["PGRIS"].pipelines["Final"].compileOptions.numPayloadValues = 8;
            m_TracerMap["PGRIS"].pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["PGRIS"].pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["PGRIS"].pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["PGRIS"].pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["PGRIS"].pipelines["Final"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_FINAL | PARAM_FLAG_NEE | PARAM_FLAG_RIS;
            if (m_EnableTree) {
                flags |= PARAM_FLAG_USE_TREE;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["PGRIS"].pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGRIS"].pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["PGRIS"].pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Trace","Build","Final" };
    for (auto& stName : stNames) {
        m_TracerMap["PGRIS"].pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        m_TracerMap["PGRIS"].pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        m_TracerMap["PGRIS"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        m_TracerMap["PGRIS"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        m_TracerMap["PGRIS"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["PGRIS"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["PGRIS"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        m_TracerMap["PGRIS"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    m_TracerMap["PGRIS"].pipelines["Build"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGRIS"].pipelines["Build"].shaderTable = this->NewShaderTable();

    m_TracerMap["PGRIS"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGRIS"].pipelines["Trace"].shaderTable = this->NewShaderTable();

    m_TracerMap["PGRIS"].pipelines["Final"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["PGRIS"].pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    for (auto& stName : stNames) {
        m_TracerMap["PGRIS"].pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        m_TracerMap["PGRIS"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
        m_TracerMap["PGRIS"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        m_TracerMap["PGRIS"].pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
        {
            auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
            auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
            auto geometryAS = instanceData->GetDwGeometryAS();
            if (geometryAS)
            {
                for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                {
                    if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                    {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes)
                        {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                            auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupData = HitgroupData();
                                if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                                {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                    hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                    hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                    hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                    hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                    hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                    hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                    hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                    hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                    hitgroupData.shinness = material.GetFloat1("shinness");
                                    hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                    auto diffTexStr = material.GetString("diffTex");
                                    if (diffTexStr == "")
                                    {
                                        diffTexStr = "White";
                                    }
                                    auto specTexStr = material.GetString("specTex");
                                    if (specTexStr == "")
                                    {
                                        specTexStr = "White";
                                    }
                                    auto emitTexStr = material.GetString("emitTex");
                                    if (emitTexStr == "")
                                    {
                                        emitTexStr = "White";
                                    }
                                    hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                    hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                    hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                                }
                                m_TracerMap["PGRIS"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                            }
                        }
                    }
                    if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                        auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                        auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = sphereRes->materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["PGRIS"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                        }
                    }
                }
            }
        }
    }

    m_TracerMap["PGRIS"].pipelines["Build"].shaderTable->Upload();
    m_TracerMap["PGRIS"].pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["PGRIS"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["PGRIS"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["PGRIS"].pipelines["Final"].shaderTable->Upload();
    m_TracerMap["PGRIS"].pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["PGRIS"].InitParams(m_Opx7Context.get(), params);
}

void RTLibExtOPX7TestApplication::InitHashTreeDefTracer()
{
    {
        //Locate
        {
            m_TracerMap["HTDEF"].pipelines["Locate"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTDEF"].pipelines["Locate"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTDEF"].pipelines["Locate"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTDEF"].pipelines["Locate"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTDEF"].pipelines["Locate"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTDEF"].pipelines["Locate"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTDEF"].pipelines["Locate"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTDEF"].pipelines["Locate"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTDEF"].pipelines["Locate"].linkOptions.maxTraceDepth = 1;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE|PARAM_FLAG_LOCATE;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTDEF"].pipelines["Locate"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTDEF"].pipelines["Locate"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTDEF"].pipelines["Locate"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Build
        {
            m_TracerMap["HTDEF"].pipelines["Build"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTDEF"].pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTDEF"].pipelines["Build"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTDEF"].pipelines["Build"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTDEF"].pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTDEF"].pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTDEF"].pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTDEF"].pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTDEF"].pipelines["Build"].linkOptions.maxTraceDepth = 1;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTDEF"].pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTDEF"].pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTDEF"].pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            m_TracerMap["HTDEF"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTDEF"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTDEF"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTDEF"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTDEF"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTDEF"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTDEF"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTDEF"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTDEF"].pipelines["Trace"].linkOptions.maxTraceDepth = 1;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTDEF"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTDEF"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTDEF"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            m_TracerMap["HTDEF"].pipelines["Final"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTDEF"].pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTDEF"].pipelines["Final"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTDEF"].pipelines["Final"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTDEF"].pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTDEF"].pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTDEF"].pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTDEF"].pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTDEF"].pipelines["Final"].linkOptions.maxTraceDepth = 1;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_FINAL;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTDEF"].pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTDEF"].pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTDEF"].pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Locate","Trace","Build","Final" };
    for (auto& stName : stNames) {
        m_TracerMap["HTDEF"].pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        m_TracerMap["HTDEF"].pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        m_TracerMap["HTDEF"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        m_TracerMap["HTDEF"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        m_TracerMap["HTDEF"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["HTDEF"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["HTDEF"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        m_TracerMap["HTDEF"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }

    m_TracerMap["HTDEF"].pipelines["Locate"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTDEF"].pipelines["Locate"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTDEF"].pipelines["Build"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTDEF"].pipelines["Build"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTDEF"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTDEF"].pipelines["Trace"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTDEF"].pipelines["Final"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTDEF"].pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    for (auto& stName : stNames) {
        m_TracerMap["HTDEF"].pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        m_TracerMap["HTDEF"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
        m_TracerMap["HTDEF"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        m_TracerMap["HTDEF"].pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
        {
            auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
            auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
            auto geometryAS = instanceData->GetDwGeometryAS();
            if (geometryAS)
            {
                for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                {
                    if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                    {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes)
                        {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                            auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupData = HitgroupData();
                                if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                                {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                    hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                    hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                    hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                    hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                    hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                    hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                    hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                    hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                    hitgroupData.shinness = material.GetFloat1("shinness");
                                    hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                    auto diffTexStr = material.GetString("diffTex");
                                    if (diffTexStr == "")
                                    {
                                        diffTexStr = "White";
                                    }
                                    auto specTexStr = material.GetString("specTex");
                                    if (specTexStr == "")
                                    {
                                        specTexStr = "White";
                                    }
                                    auto emitTexStr = material.GetString("emitTex");
                                    if (emitTexStr == "")
                                    {
                                        emitTexStr = "White";
                                    }
                                    hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                    hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                    hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                                }
                                m_TracerMap["HTDEF"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                            }
                        }
                    }
                    if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                        auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                        auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = sphereRes->materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["HTDEF"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                        }
                    }
                }
            }
        }
    }

    m_TracerMap["HTDEF"].pipelines["Locate"].shaderTable->Upload();
    m_TracerMap["HTDEF"].pipelines["Locate"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["HTDEF"].pipelines["Build"].shaderTable->Upload();
    m_TracerMap["HTDEF"].pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["HTDEF"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["HTDEF"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["HTDEF"].pipelines["Final"].shaderTable->Upload();
    m_TracerMap["HTDEF"].pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["HTDEF"].InitParams(m_Opx7Context.get(), params);
}

void RTLibExtOPX7TestApplication::InitHashTreeNeeTracer()
{
    {
        //Locate
        {
            m_TracerMap["HTNEE"].pipelines["Locate"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTNEE"].pipelines["Locate"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTNEE"].pipelines["Locate"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTNEE"].pipelines["Locate"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTNEE"].pipelines["Locate"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTNEE"].pipelines["Locate"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTNEE"].pipelines["Locate"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTNEE"].pipelines["Locate"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTNEE"].pipelines["Locate"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_NEE | PARAM_FLAG_LOCATE;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTNEE"].pipelines["Locate"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTNEE"].pipelines["Locate"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTNEE"].pipelines["Locate"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Build
        {
            m_TracerMap["HTNEE"].pipelines["Build"].compileOptions.usesMotionBlur            = false;
            m_TracerMap["HTNEE"].pipelines["Build"].compileOptions.traversableGraphFlags     = 0;
            m_TracerMap["HTNEE"].pipelines["Build"].compileOptions.numAttributeValues        = 3;
            m_TracerMap["HTNEE"].pipelines["Build"].compileOptions.numPayloadValues          = 8;
            m_TracerMap["HTNEE"].pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTNEE"].pipelines["Build"].compileOptions.usesPrimitiveTypeFlags    = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTNEE"].pipelines["Build"].compileOptions.exceptionFlags            = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTNEE"].pipelines["Build"].linkOptions.debugLevel                   = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTNEE"].pipelines["Build"].linkOptions.maxTraceDepth                = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_NEE;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTNEE"].pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTNEE"].pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTNEE"].pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            m_TracerMap["HTNEE"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTNEE"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTNEE"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTNEE"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTNEE"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTNEE"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTNEE"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTNEE"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTNEE"].pipelines["Trace"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_NEE;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTNEE"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTNEE"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTNEE"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            m_TracerMap["HTNEE"].pipelines["Final"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTNEE"].pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTNEE"].pipelines["Final"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTNEE"].pipelines["Final"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTNEE"].pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTNEE"].pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTNEE"].pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTNEE"].pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTNEE"].pipelines["Final"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_FINAL | PARAM_FLAG_NEE;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTNEE"].pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTNEE"].pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTNEE"].pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Locate","Trace","Build","Final"};
    for (auto& stName : stNames) {
        m_TracerMap["HTNEE"].pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        m_TracerMap["HTNEE"].pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        m_TracerMap["HTNEE"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        m_TracerMap["HTNEE"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        m_TracerMap["HTNEE"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["HTNEE"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["HTNEE"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        m_TracerMap["HTNEE"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    m_TracerMap["HTNEE"].pipelines["Locate"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTNEE"].pipelines["Locate"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTNEE"].pipelines["Build"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTNEE"].pipelines["Build"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTNEE"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTNEE"].pipelines["Trace"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTNEE"].pipelines["Final"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTNEE"].pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    for (auto& stName : stNames) {
        m_TracerMap["HTNEE"].pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        m_TracerMap["HTNEE"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
        m_TracerMap["HTNEE"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        m_TracerMap["HTNEE"].pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
        {
            auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
            auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
            auto geometryAS = instanceData->GetDwGeometryAS();
            if (geometryAS)
            {
                for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                {
                    if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                    {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes)
                        {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                            auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupData = HitgroupData();
                                if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                                {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                    hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                    hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                    hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                    hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                    hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                    hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                    hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                    hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                    hitgroupData.shinness = material.GetFloat1("shinness");
                                    hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                    auto diffTexStr = material.GetString("diffTex");
                                    if (diffTexStr == "")
                                    {
                                        diffTexStr = "White";
                                    }
                                    auto specTexStr = material.GetString("specTex");
                                    if (specTexStr == "")
                                    {
                                        specTexStr = "White";
                                    }
                                    auto emitTexStr = material.GetString("emitTex");
                                    if (emitTexStr == "")
                                    {
                                        emitTexStr = "White";
                                    }
                                    hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                    hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                    hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                                }
                                m_TracerMap["HTNEE"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                            }
                        }
                    }
                    if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                        auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                        auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = sphereRes->materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["HTNEE"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                        }
                    }
                }
            }
        }
    }

    m_TracerMap["HTNEE"].pipelines["Locate"].shaderTable->Upload();
    m_TracerMap["HTNEE"].pipelines["Locate"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    m_TracerMap["HTNEE"].pipelines["Build" ].shaderTable->Upload();
    m_TracerMap["HTNEE"].pipelines["Build" ].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    m_TracerMap["HTNEE"].pipelines["Trace" ].shaderTable->Upload();
    m_TracerMap["HTNEE"].pipelines["Trace" ].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    m_TracerMap["HTNEE"].pipelines["Final" ].shaderTable->Upload();
    m_TracerMap["HTNEE"].pipelines["Final" ].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["HTNEE"].InitParams(m_Opx7Context.get(), params);
}

void RTLibExtOPX7TestApplication::InitHashTreeRisTracer()
{
    {
        //Locate
        {
            m_TracerMap["HTRIS"].pipelines["Locate"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTRIS"].pipelines["Locate"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTRIS"].pipelines["Locate"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTRIS"].pipelines["Locate"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTRIS"].pipelines["Locate"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTRIS"].pipelines["Locate"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTRIS"].pipelines["Locate"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTRIS"].pipelines["Locate"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTRIS"].pipelines["Locate"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_NEE | PARAM_FLAG_RIS | PARAM_FLAG_LOCATE;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTRIS"].pipelines["Locate"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTRIS"].pipelines["Locate"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTRIS"].pipelines["Locate"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Build
        {
            m_TracerMap["HTRIS"].pipelines["Build"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTRIS"].pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTRIS"].pipelines["Build"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTRIS"].pipelines["Build"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTRIS"].pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTRIS"].pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTRIS"].pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTRIS"].pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTRIS"].pipelines["Build"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_NEE | PARAM_FLAG_RIS;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTRIS"].pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTRIS"].pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTRIS"].pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            m_TracerMap["HTRIS"].pipelines["Trace"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTRIS"].pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTRIS"].pipelines["Trace"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTRIS"].pipelines["Trace"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTRIS"].pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTRIS"].pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTRIS"].pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTRIS"].pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTRIS"].pipelines["Trace"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_NEE | PARAM_FLAG_RIS;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTRIS"].pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTRIS"].pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTRIS"].pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            m_TracerMap["HTRIS"].pipelines["Final"].compileOptions.usesMotionBlur = false;
            m_TracerMap["HTRIS"].pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            m_TracerMap["HTRIS"].pipelines["Final"].compileOptions.numAttributeValues = 3;
            m_TracerMap["HTRIS"].pipelines["Final"].compileOptions.numPayloadValues = 8;
            m_TracerMap["HTRIS"].pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            m_TracerMap["HTRIS"].pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            m_TracerMap["HTRIS"].pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            m_TracerMap["HTRIS"].pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            m_TracerMap["HTRIS"].pipelines["Final"].linkOptions.maxTraceDepth = 2;

            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
            unsigned int flags = PARAM_FLAG_NONE | PARAM_FLAG_BUILD | PARAM_FLAG_FINAL | PARAM_FLAG_NEE | PARAM_FLAG_RIS;
            if (m_EnableGrid) {
                flags |= PARAM_FLAG_USE_GRID;
            }

#ifdef NDEBUG
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
            moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
            moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
#endif
            moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleOptions.payloadTypes = {};
            moduleOptions.boundValueEntries.push_back({});
            moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
            moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
            moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
            m_TracerMap["HTRIS"].pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            m_TracerMap["HTRIS"].pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            m_TracerMap["HTRIS"].pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Locate","Trace","Build","Final" };
    for (auto& stName : stNames) {
        m_TracerMap["HTRIS"].pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        m_TracerMap["HTRIS"].pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        m_TracerMap["HTRIS"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        m_TracerMap["HTRIS"].pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        m_TracerMap["HTRIS"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["HTRIS"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        m_TracerMap["HTRIS"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        m_TracerMap["HTRIS"].pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    m_TracerMap["HTRIS"].pipelines["Locate"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTRIS"].pipelines["Locate"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTRIS"].pipelines["Build"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTRIS"].pipelines["Build"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTRIS"].pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTRIS"].pipelines["Trace"].shaderTable = this->NewShaderTable();

    m_TracerMap["HTRIS"].pipelines["Final"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["HTRIS"].pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    for (auto& stName : stNames) {
        m_TracerMap["HTRIS"].pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        m_TracerMap["HTRIS"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
        m_TracerMap["HTRIS"].pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        m_TracerMap["HTRIS"].pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
        {
            auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
            auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
            auto geometryAS = instanceData->GetDwGeometryAS();
            if (geometryAS)
            {
                for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                {
                    if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
                    {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes)
                        {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                            auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupData = HitgroupData();
                                if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                                {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupData.type = rtlib::test::SpecifyMaterialType(material,mesh);
                                    hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                    hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                    hitgroupData.indCount = extUniqueData->GetTriIdxCount();
                                    hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                    hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                    hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                    hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                    hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                    hitgroupData.shinness = material.GetFloat1("shinness");
                                    hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                    auto diffTexStr = material.GetString("diffTex");
                                    if (diffTexStr == "")
                                    {
                                        diffTexStr = "White";
                                    }
                                    auto specTexStr = material.GetString("specTex");
                                    if (specTexStr == "")
                                    {
                                        specTexStr = "White";
                                    }
                                    auto emitTexStr = material.GetString("emitTex");
                                    if (emitTexStr == "")
                                    {
                                        emitTexStr = "White";
                                    }
                                    hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                    hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                    hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                                }
                                m_TracerMap["HTRIS"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
                            }
                        }
                    }
                    if (m_SceneData.world.geometrySpheres.count(geometryName) > 0) {
                        auto& sphereRes = m_SceneData.sphereResources.at(geometryName);
                        auto  desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + geometryName);
                        for (auto i = 0; i < desc.recordCount; ++i)
                        {
                            auto hitgroupData = HitgroupData();
                            if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                            {
                                auto material = sphereRes->materials[i / desc.recordStride];
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>()->GetCenterBufferGpuAddress());
                                hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                hitgroupData.shinness = material.GetFloat1("shinness");
                                hitgroupData.refIndex = material.GetFloat1("refrIndx");
                                auto diffTexStr = material.GetString("diffTex");
                                if (diffTexStr == "")
                                {
                                    diffTexStr = "White";
                                }
                                auto specTexStr = material.GetString("specTex");
                                if (specTexStr == "")
                                {
                                    specTexStr = "White";
                                }
                                auto emitTexStr = material.GetString("emitTex");
                                if (emitTexStr == "")
                                {
                                    emitTexStr = "White";
                                }
                                hitgroupData.diffuseTex = m_TextureMap.at(diffTexStr).GetCUtexObject();
                                hitgroupData.specularTex = m_TextureMap.at(emitTexStr).GetCUtexObject();
                                hitgroupData.emissionTex = m_TextureMap.at(specTexStr).GetCUtexObject();

                            }
                            m_TracerMap["HTRIS"].pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                        }
                    }
                }
            }
        }
    }

    m_TracerMap["HTRIS"].pipelines["Locate"].shaderTable->Upload();
    m_TracerMap["HTRIS"].pipelines["Locate"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["HTRIS"].pipelines["Build"].shaderTable->Upload();
    m_TracerMap["HTRIS"].pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["HTRIS"].pipelines["Trace"].shaderTable->Upload();
    m_TracerMap["HTRIS"].pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    m_TracerMap["HTRIS"].pipelines["Final"].shaderTable->Upload();
    m_TracerMap["HTRIS"].pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["HTRIS"].InitParams(m_Opx7Context.get(), params);
}

 void RTLibExtOPX7TestApplication::FreeTracers()
{
    for (auto& [name, pipeline] : m_TracerMap)
    {
        pipeline.Free();
    }
    m_TracerMap.clear();
}

 void RTLibExtOPX7TestApplication::InitFrameResourceCUDA()
{
    size_t pixelSize = m_SceneData.config.width * m_SceneData.config.height;
    m_AccumBufferCUDA = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(m_Opx7Context->CreateBuffer({ RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, pixelSize * sizeof(float) * 3, nullptr }));
    m_FrameBufferCUDA = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(m_Opx7Context->CreateBuffer({ RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, pixelSize * sizeof(uchar4), nullptr }));
    auto rnd = std::random_device();
    auto mt19937 = std::mt19937(rnd());
    auto seedData = std::vector<unsigned int>(pixelSize * sizeof(unsigned int));
    std::generate(std::begin(seedData), std::end(seedData), mt19937);
    m_SeedBufferCUDA = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(m_Opx7Context->CreateBuffer(
        { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, seedData.size() * sizeof(seedData[0]), seedData.data() }));
}

 void RTLibExtOPX7TestApplication::FreeFrameResourceCUDA()
{
    if (m_AccumBufferCUDA)
    {
        m_AccumBufferCUDA->Destroy();
        m_AccumBufferCUDA = nullptr;
    }
    if (m_FrameBufferCUDA)
    {
        m_FrameBufferCUDA->Destroy();
        m_FrameBufferCUDA = nullptr;
    }
    if (m_SeedBufferCUDA)
    {
        m_SeedBufferCUDA->Destroy();
        m_SeedBufferCUDA = nullptr;
    }
}

 void RTLibExtOPX7TestApplication::InitFrameResourceCUGL()
{
    m_FrameBufferCUGL = std::unique_ptr<RTLib::Ext::CUGL::CUGLBuffer>(RTLib::Ext::CUGL::CUGLBuffer::New(m_Opx7Context.get(), m_FrameBufferGL.get(), RTLib::Ext::CUGL::CUGLGraphicsRegisterFlagsWriteDiscard));
}

 void RTLibExtOPX7TestApplication::FreeFrameResourceCUGL()
{
    if (m_FrameBufferCUGL)
    {
        m_FrameBufferCUGL->Destroy();
        m_FrameBufferCUGL.reset();
    }
}

 void RTLibExtOPX7TestApplication::InitFrameResourceOGL4()
{
    size_t pixelSize = m_SceneData.config.width * m_SceneData.config.height;
    auto ogl4Context = m_GlfwWindow->GetOpenGLContext();
    m_FrameBufferGL = std::unique_ptr<RTLib::Ext::GL::GLBuffer>(ogl4Context->CreateBuffer(RTLib::Ext::GL::GLBufferCreateDesc{ sizeof(uchar4) * pixelSize, RTLib::Ext::GL::GLBufferUsageImageCopySrc, RTLib::Ext::GL::GLMemoryPropertyDefault, nullptr }));
    m_FrameTextureGL = rtlib::test::CreateFrameTextureGL(ogl4Context, m_SceneData.config.width, m_SceneData.config.height);
}

 void RTLibExtOPX7TestApplication::FreeFrameResourceOGL4()
{
    if (m_FrameBufferGL)
    {
        m_FrameBufferGL->Destroy();
        m_FrameBufferGL = nullptr;
    }
    if (m_FrameTextureGL)
    {
        m_FrameTextureGL->Destroy();
        m_FrameTextureGL = nullptr;
    }
}

 void RTLibExtOPX7TestApplication::InitRectRendererGL()
{
    auto ogl4Context = m_GlfwWindow->GetOpenGLContext();
    m_RectRendererGL = std::unique_ptr<RTLib::Ext::GL::GLRectRenderer>(ogl4Context->CreateRectRenderer({ 1, false, true }));
}

 void RTLibExtOPX7TestApplication::FreeRectRendererGL()
{
    if (m_RectRendererGL)
    {
        m_RectRendererGL->Destroy();
        m_RectRendererGL.reset();
    }
}

 void RTLibExtOPX7TestApplication::InitWindowCallback()
{
    m_GlfwWindow->SetResizable(true);
    m_GlfwWindow->SetUserPointer(&m_WindowState);
    m_GlfwWindow->SetCursorPosCallback(CursorPosCallback);
}

 bool RTLibExtOPX7TestApplication::TracePipeline(RTLib::Ext::CUDA::CUDAStream* stream, RTLib::Ext::CUDA::CUDABuffer* frameBuffer)
{   
    m_PipelineName = "Trace";
    if ((m_CurTracerName == "PGDEF") || (m_CurTracerName == "PGNEE") || (m_CurTracerName == "PGRIS"))
    {
        m_SdTreeController->BegTrace(stream);
        if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecord)
        {
            m_PipelineName = "Build";
        }
        if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecordAndSample) {
            m_PipelineName = "Trace";
        }
        if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateSample) {
            m_PipelineName = "Final";
        }
    }
    if ((m_CurTracerName == "HTDEF") || (m_CurTracerName == "HTNEE") || (m_CurTracerName == "HTRIS"))
    {
        m_MortonQuadTreeController->BegTrace(stream);
        if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateLocate)
        {
            m_PipelineName = "Locate";
        }
        if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecord)
        {
            m_PipelineName = "Build";
        }
        if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecordAndSample) {
            m_PipelineName = "Trace";
        }
        if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateSample) {
            m_PipelineName = "Final";
        }
    }

    auto params = GetParams(frameBuffer);
    if (stream) {
        stream->CopyMemoryToBuffer(m_TracerMap[m_CurTracerName].paramsBuffer.get(), { { &params, 0, sizeof(params) } });
    }
    else {
        m_Opx7Context->CopyMemoryToBuffer(m_TracerMap[m_CurTracerName].paramsBuffer.get(), { { &params, 0, sizeof(params) } });
    }
    m_TracerMap[m_CurTracerName].Launch(stream, m_PipelineName, m_SceneData.config.width, m_SceneData.config.height);bool shouldSync = false;
    if ((m_CurTracerName == "DEF")   || (m_CurTracerName == "NEE")   || (m_CurTracerName == "RIS")) {
        if (m_EnableGrid) {
            m_HashBufferCUDA.Update(m_Opx7Context.get(),stream);
            if (stream) {
                shouldSync = true;
            }
        }
    }
    if ((m_CurTracerName == "PGDEF") || (m_CurTracerName == "PGNEE") || (m_CurTracerName == "PGRIS")) {
        if (m_EnableTree) {
            m_SdTreeController->EndTrace(stream);
            if (stream) {
                shouldSync = true;
            }
        }
    }
    if ((m_CurTracerName == "HTDEF") || (m_CurTracerName == "HTNEE") || (m_CurTracerName == "HTRIS")) {
        m_MortonQuadTreeController->EndTrace(stream);
        if (m_EnableGrid) {
            if (m_MortonQuadTreeController->m_SamplePerTmp == 0)
            {
                m_HashBufferCUDA.Update(m_Opx7Context.get(), stream);
            }
        }
        if (stream) {
            shouldSync = true;
        }
    }
    if ( m_CurTracerName != "DBG") {
        m_SamplesForAccum += m_SceneData.config.samples;
    }
    return shouldSync;
}
 void RTLibExtOPX7TestApplication::SetupPipeline()
 {
     auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
     desc.sizeInBytes = m_FrameBufferCUDA->GetSizeInBytes();
     auto frameBufferTmp = m_Opx7Context->CreateBuffer(desc);
     TracePipeline(nullptr, frameBufferTmp);
     m_SamplesForAccum = 0;
     cuMemsetD32(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()), m_AccumBufferCUDA->GetSizeInBytes() / sizeof(float), 0);
     if (m_EnableGrid) {
         m_HashBufferCUDA.Clear(m_Opx7Context.get());
     }
     frameBufferTmp->Destroy();
     delete frameBufferTmp;
 }

 bool RTLibExtOPX7TestApplication::FinishTrace()
{
    if (m_EnableVis)
    {
        return m_GlfwWindow->ShouldClose() || (m_SamplesForAccum >= m_SceneData.config.maxSamples || (m_TimesForAccum >= m_SceneData.config.maxTimes*(1000*1000)));
    }
    else
    {
        return (m_SamplesForAccum >= m_SceneData.config.maxSamples) || (m_TimesForAccum >= m_SceneData.config.maxTimes*(1000*1000));
    }
}

 void RTLibExtOPX7TestApplication::UpdateTrace()
{
    {
        if (m_EventState.isResized)
        {
            if (m_EnableVis)
            {
                this->FreeFrameResourceCUDA();
                this->FreeFrameResourceOGL4();
                this->FreeFrameResourceCUGL();
                this->InitFrameResourceCUDA();
                this->InitFrameResourceOGL4();
                this->InitFrameResourceCUGL();
            }
            else
            {
                this->FreeFrameResourceCUDA();
                this->InitFrameResourceCUDA();
            }
        }
        if (m_EventState.isClearFrame)
        {
            auto zeroClearData = std::vector<float>(m_SceneData.config.width * m_SceneData.config.height * 3, 0.0f);
            m_Opx7Context->CopyMemoryToBuffer(m_AccumBufferCUDA.get(), { { zeroClearData.data(), 0, sizeof(zeroClearData[0]) * zeroClearData.size() } });
            m_SamplesForAccum = 0;
            m_TimesForAccum   = 0;
            this->UpdateTimeStamp();
        }
        if (m_EventState.isMovedCamera)
        {
            m_TracerMap[m_CurTracerName].pipelines[m_PipelineName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
            m_TracerMap[m_CurTracerName].pipelines[m_PipelineName].shaderTable->UploadRaygenRecord();
        }
        if (m_SamplesForAccum % 100 == 99)
        {
            auto rnd = std::random_device();
            auto mt19937 = std::mt19937(rnd());
            auto seedData = std::vector<unsigned int>(m_SceneData.config.width * m_SceneData.config.height * sizeof(unsigned int));
            std::generate(std::begin(seedData), std::end(seedData), mt19937);
            m_Opx7Context->CopyMemoryToBuffer(m_SeedBufferCUDA.get(), { { seedData.data(), 0, sizeof(seedData[0]) * std::size(seedData) } });
        }
    }
}

 void RTLibExtOPX7TestApplication::UpdateState()
{
     m_EventState = rtlib::test::EventState();
    if (m_EnableVis)
    {

        m_GlfwContext->Update();
        m_KeyBoardManager->Update();
        m_MouseButtonManager->Update();
        float prevTime = glfwGetTime();
        {
            m_WindowState.delTime = m_WindowState.curTime - prevTime;
            m_WindowState.curTime = prevTime;
        }
        {
            auto tmpSize = m_GlfwWindow->GetSize();
            if (m_SceneData.config.width != tmpSize.width || m_SceneData.config.height != tmpSize.height)
            {
                std::cout << m_SceneData.config.width << "->" << tmpSize.width << "\n";
                std::cout << m_SceneData.config.height << "->" << tmpSize.height << "\n";
                m_SceneData.config.width = tmpSize.width;
                m_SceneData.config.height = tmpSize.height;
                m_EventState.isResized = true;
            }
            else
            {
                m_EventState.isResized = false;
            }
        }
        m_EventState.isMovedCamera = rtlib::test::UpdateCameraMovement(
            m_KeyBoardManager.get(),
            m_MouseButtonManager.get(),
            m_SceneData.cameraController,
            m_WindowState.delTime,
            m_WindowState.delCurPos.x,
            m_WindowState.delCurPos.y);
        if (m_KeyBoardManager->GetState(GLFW_KEY_F1)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F1)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "DBG")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "DBG";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F2)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F2)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "DEF")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "DEF";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F3)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F3)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "NEE")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "NEE";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F4)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F4)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "RIS")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "RIS";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F5)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F5)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "PGDEF")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName  = "PGDEF";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
                m_SdTreeController->Start();
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F6)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F6)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "PGNEE")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "PGNEE";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
                m_SdTreeController->Start();
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F7)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F7)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "PGRIS")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "PGRIS";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
                m_SdTreeController->Start();
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F8)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F8)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "HTDEF")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "HTDEF";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
                m_MortonQuadTreeController->Start();
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F9)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F9)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "HTNEE")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "HTNEE";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
                m_MortonQuadTreeController->Start();
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F10)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F10)->isPressed)
        {
            m_PrvTracerName = m_CurTracerName;
            if (m_PrvTracerName != "HTRIS")
            {
                std::cout << "State Change" << std::endl;
                m_CurTracerName = "HTRIS";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
                m_MortonQuadTreeController->Start();
            }
        }

        if (m_CurTracerName == "DBG") {
            for (int i = 0; i < DEBUG_FRAME_TYPE_COUNT; ++i) {
                if ((m_KeyBoardManager->GetState(GLFW_KEY_1 + i)->isPressed &&
                     m_KeyBoardManager->GetState(GLFW_KEY_1 + i)->isUpdated)|| 
                    (m_KeyBoardManager->GetState(GLFW_KEY_KP_1 + i)->isPressed &&
                     m_KeyBoardManager->GetState(GLFW_KEY_KP_1 + i)->isUpdated)
                    ) {
                    m_DebugFrameType = i + 1;
                }
            }
        }
        if (m_EventState.isResized)
        {
            m_EventState.isMovedCamera = true;
            m_EventState.isClearFrame = true;
        }
        if (m_EventState.isMovedCamera)
        {
            m_EventState.isClearFrame = true;
        }
        m_GlfwWindow->SwapBuffers();
    }

}

 void RTLibExtOPX7TestApplication::UpdateTimeStamp()
{
    time_t now = std::time(nullptr);
    auto t = std::localtime(&now);
    char str[256];
    std::strftime(str, sizeof(str), "%x-%X", t);
    m_TimeStampString = std::string(str);
    std::replace(m_TimeStampString.begin(), m_TimeStampString.end(), '/', '-');
    std::replace(m_TimeStampString.begin(), m_TimeStampString.end(), ':', '-');
}

 void RTLibExtOPX7TestApplication::TraceFrame(RTLib::Ext::CUDA::CUDAStream* stream)
{
     if (stream) {
         RTLIB_CORE_ASSERT_IF_FAILED(stream->Synchronize());
     }
    if (m_EnableVis)
    {
        auto  beg = std::chrono::system_clock::now();
        auto frameBufferCUDA = m_FrameBufferCUGL->Map(stream);
        (void)this->TracePipeline(stream,  frameBufferCUDA);
        m_FrameBufferCUGL->Unmap(stream);
        if (stream) {
            RTLIB_CORE_ASSERT_IF_FAILED(stream->Synchronize());
        }
        auto end = std::chrono::system_clock::now();
        m_TimesForFrame = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
        if (m_EventState.isResized)
        {
            glViewport(0, 0, m_SceneData.config.width, m_SceneData.config.height);
        }
        rtlib::test::RenderFrameGL(m_GlfwWindow->GetOpenGLContext(), m_RectRendererGL.get(), m_FrameBufferGL.get(), m_FrameTextureGL.get());
    }
    else
    {
        auto beg = std::chrono::system_clock::now();
        if (this->TracePipeline(stream, m_FrameBufferCUDA.get())) {
            RTLIB_CORE_ASSERT_IF_FAILED(stream->Synchronize());
        }
        auto end = std::chrono::system_clock::now();
        m_TimesForFrame = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
    }
    if (m_CurTracerName != "DBG") {
        m_TimesForAccum += m_TimesForFrame;
        if ((m_SamplesForAccum > 0) && (m_SamplesForAccum % m_SceneData.config.samplesPerSave == 0))
        {
            SaveResultImage(stream);
        }
    }
}

 void RTLibExtOPX7TestApplication::SaveResultImage(RTLib::Ext::CUDA::CUDAStream* stream)
 {
     auto baseSavePath = std::filesystem::path(m_SceneData.config.imagePath).make_preferred() / m_TimeStampString;
     if (!std::filesystem::exists(baseSavePath))
     {
         std::filesystem::create_directories(baseSavePath);
         std::filesystem::copy_file(m_ScenePath, baseSavePath / "scene.json");
     }
     auto configData = rtlib::test::ImageConfigData();
     configData.width = m_SceneData.config.width;
     configData.height = m_SceneData.config.height;
     configData.samples = m_SamplesForAccum;
     configData.time = m_TimesForAccum/(1000*1000);
     configData.enableVis = m_EnableVis;
     configData.pngFilePath = baseSavePath.string() + "/result_" + m_CurTracerName + "_" + std::to_string(m_SamplesForAccum) + ".png";
     configData.binFilePath = baseSavePath.string() + "/result_" + m_CurTracerName + "_" + std::to_string(m_SamplesForAccum) + ".bin";
     configData.exrFilePath = baseSavePath.string() + "/result_" + m_CurTracerName + "_" + std::to_string(m_SamplesForAccum) + ".exr";
     {
         std::ofstream configFile(baseSavePath.string() + "/config_" + m_CurTracerName + "_" + std::to_string(m_SamplesForAccum) + ".json");
         configFile << nlohmann::json(configData);
         configFile.close();
     }

     auto pixelSize = m_SceneData.config.width * m_SceneData.config.height;
     auto download_data = std::vector<float>(pixelSize * 3, 0.0f);
     {
         auto memoryCopy = RTLib::Ext::CUDA::CUDABufferMemoryCopy();
         memoryCopy.dstData = download_data.data();
         memoryCopy.srcOffset = 0;
         memoryCopy.size = pixelSize * sizeof(float) * 3;
         RTLIB_CORE_ASSERT_IF_FAILED(m_Opx7Context->CopyBufferToMemory(m_AccumBufferCUDA.get(), { memoryCopy }));
     }
     {
         auto hdr_image_data = std::vector<float>(pixelSize * 3, 0.0f);
         for (size_t i = 0; i < pixelSize; ++i)
         {
             hdr_image_data[3 * (pixelSize - 1 - i) + 0] = download_data[3 * i + 0] / static_cast<float>(m_SamplesForAccum);
             hdr_image_data[3 * (pixelSize - 1 - i) + 1] = download_data[3 * i + 1] / static_cast<float>(m_SamplesForAccum);
             hdr_image_data[3 * (pixelSize - 1 - i) + 2] = download_data[3 * i + 2] / static_cast<float>(m_SamplesForAccum);
         }
         rtlib::test::SaveExrImage(configData.exrFilePath.c_str(), m_SceneData.config.width, m_SceneData.config.height, hdr_image_data);
         std::ofstream imageBinFile(configData.binFilePath, std::ios::binary | std::ios::ate);
         imageBinFile.write((char*)hdr_image_data.data(), hdr_image_data.size() * sizeof(hdr_image_data[0]));
         imageBinFile.close();
     }
     {
         auto png_image_data = std::vector<unsigned char>(pixelSize * 4);
         for (size_t i = 0; i < pixelSize; ++i)
         {
             png_image_data[4 * i + 0] = 255.99f * std::min(RTLib::Ext::CUDA::Math::linear_to_gamma(download_data[3 * (pixelSize - 1 - i) + 0] / static_cast<float>(m_SamplesForAccum)), 1.0f);
             png_image_data[4 * i + 1] = 255.99f * std::min(RTLib::Ext::CUDA::Math::linear_to_gamma(download_data[3 * (pixelSize - 1 - i) + 1] / static_cast<float>(m_SamplesForAccum)), 1.0f);
             png_image_data[4 * i + 2] = 255.99f * std::min(RTLib::Ext::CUDA::Math::linear_to_gamma(download_data[3 * (pixelSize - 1 - i) + 2] / static_cast<float>(m_SamplesForAccum)), 1.0f);
             png_image_data[4 * i + 3] = 255;
         }
         rtlib::test::SavePngImage(configData.pngFilePath.c_str(), m_SceneData.config.width, m_SceneData.config.height, png_image_data);
     }
 }