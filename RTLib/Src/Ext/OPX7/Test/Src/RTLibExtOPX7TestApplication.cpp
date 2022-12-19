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
    auto tmpBgColor = m_SceneData.config.custom.GetFloat3Or("MissData.BgColor", { 0.0f ,0.0f ,0.0f });
    m_BgColor = {tmpBgColor[0], tmpBgColor[1], tmpBgColor[2]};
    std::cout << "OK_HERE" << std::endl;
    if (argc > 1) {
        for (int i = 0; i < argc-1; ++i) {
            //if (std::string(argv[i]) == "--EnableVis") {
            //    m_EnableVis = false;
            //    if ((std::string(argv[i + 1]) == "true") || (std::string(argv[i + 1]) == "True") ||
            //        (std::string(argv[i + 1]) == "on") || (std::string(argv[i + 1]) == "On") ||
            //        (std::string(argv[i + 1]) == "1")) {
            //        std::cout << "SUC: --EnableVis Args is ON\n";
            //        m_EnableVis = true;
            //    }
            //    else if ((std::string(argv[i + 1]) == "false") || (std::string(argv[i + 1]) == "False") ||
            //        (std::string(argv[i + 1]) == "off") || (std::string(argv[i + 1]) == "Off") ||
            //        (std::string(argv[i + 1]) == "0")) {
            //        std::cout << "SUC: --EnableVis Args is OFF\n";
            //        m_EnableVis = false;
            //    }
            //    else {
            //        std::cout << "BUG: --EnableVis Args is Missing: Use Default(false)\n";
            //    }
            //}
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
            auto tmpAabb = rtlib::test::TransformAABB(RTLib::Core::AABB(m_GeometryASMap.at(geometryAS->GetName()).aabbMin, m_GeometryASMap.at(geometryAS->GetName()).aabbMax),rtlib::test::GetInstanceTransform(m_SceneData.world, instancePath));
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
                            auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
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
    m_PtxStringMap["SimpleTrace.ptx"]  = rtlib::test::LoadBinary<char>(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleTrace.optixir");
    m_PtxStringMap["SimpleGuide.ptx"]  = rtlib::test::LoadBinary<char>(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleGuide.optixir");
    m_PtxStringMap["SimpleGuide2.ptx"] = rtlib::test::LoadBinary<char>(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleGuide2.optixir");
}

void RTLibExtOPX7TestApplication::InitDefTracer()
{
    auto tracer = rtlib::test::TracerData();
    tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
    tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
    tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
    tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
    tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
    tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle| RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 1;
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
        tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.DEF", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.DEF", moduleOptions);
        tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(),   "BuiltIn.Sphere.DEF", moduleOptions);
    }
    tracer.pipelines["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.DEF", "__raygen__default");
    tracer.pipelines["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.DEF", "__exception__ep");
    tracer.pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.DEF", "__miss__radiance");
    tracer.pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.DEF", "__miss__occluded");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.DEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.DEF", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.DEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.DEF", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere"  , "SimpleKernel.DEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.DEF", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere"  , "SimpleKernel.DEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.DEF", "");
    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" 
    };

    tracer.pipelines["Trace"].SetHostRayGenRecordTypeData(   rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    tracer.pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{make_float4(m_BgColor,1.0f)});
    tracer.pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    tracer.pipelines["Trace"].SetHostExceptionRecordTypeData(unsigned int());

    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
    {
        if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
        {
            auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
            auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
            for (auto& [meshName, meshData] : geometry.meshes)
            {
                auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                for (auto i = 0; i < desc.recordCount; ++i)
                {
                    auto hitgroupData = HitgroupData();
                    if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                    {
                        hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                    }
                    tracer.pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                    hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                }
                tracer.pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
            }
        }
    }
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(),params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream){
            return "Trace";
        }
    );
    tracer.getParams  = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            {
                params.flags = PARAM_FLAG_NONE;

                if (m_EnableGrid) {
                    params.flags |= PARAM_FLAG_USE_GRID;
                    params.mortonTree = m_MortonQuadTree->GetGpuHandle();
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false;
            {
                if (m_EnableGrid) {
                    m_HashBufferCUDA.Update(m_Opx7Context.get(), stream);
                    if (stream) {
                        shouldSync = true;
                    }
                }
            }
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["DEF"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitNeeTracer()
{
    auto tracer = rtlib::test::TracerData();
    tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
    tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
    tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
    tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
    tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
    tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 2;
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
        tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.NEE", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.NEE", moduleOptions);
        tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.NEE", moduleOptions);
    }
    tracer.pipelines["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.NEE", "__raygen__default");
    tracer.pipelines["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.NEE", "__exception__ep");
    tracer.pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.NEE", "__miss__radiance");
    tracer.pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.NEE", "__miss__occluded");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.NEE", "__closesthit__radiance", "", "", "BuiltIn.Triangle.NEE", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.NEE", "__closesthit__occluded", "", "", "BuiltIn.Triangle.NEE", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.NEE", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.NEE", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.NEE", "__closesthit__occluded", "", "", "BuiltIn.Sphere.NEE", "");
    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" 
    };
    tracer.pipelines["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    tracer.pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{ make_float4(m_BgColor,1.0f) });
    tracer.pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    tracer.pipelines["Trace"].SetHostExceptionRecordTypeData( unsigned int());

    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
    {
        if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
        {
            auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
            auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
            for (auto& [meshName, meshData] : geometry.meshes)
            {
                auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                for (auto i = 0; i < desc.recordCount; ++i)
                {
                    auto hitgroupData = HitgroupData();
                    if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                    {
                        hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                    }
                    tracer.pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                    hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                }
                tracer.pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
            }
        }
    }
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(), params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            {
                params.flags = PARAM_FLAG_NEE;

                if (m_EnableGrid) {

                    params.flags |= PARAM_FLAG_USE_GRID;
                    params.mortonTree = m_MortonQuadTree->GetGpuHandle();
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false;
            {
                if (m_EnableGrid) {
                    m_HashBufferCUDA.Update(m_Opx7Context.get(), stream);
                    if (stream) {
                        shouldSync = true;
                    }
                }
            }
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["NEE"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitRisTracer()
{
    auto tracer = rtlib::test::TracerData();
    tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
    tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
    tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
    tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
    tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
    tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 2;
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
        tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.RIS", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.RIS", moduleOptions);
        tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.RIS", moduleOptions);
    }
    tracer.pipelines["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.RIS", "__raygen__default");
    tracer.pipelines["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.RIS", "__exception__ep");
    tracer.pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.RIS", "__miss__radiance");
    tracer.pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.RIS", "__miss__occluded");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.RIS", "__closesthit__radiance", "", "", "BuiltIn.Triangle.RIS", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.RIS", "__closesthit__occluded", "", "", "BuiltIn.Triangle.RIS", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.RIS", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.RIS", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.RIS", "__closesthit__occluded", "", "", "BuiltIn.Sphere.RIS", "");
    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" };
    tracer.pipelines["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    tracer.pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{ make_float4(m_BgColor,1.0f) });
    tracer.pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    tracer.pipelines["Trace"].SetHostExceptionRecordTypeData(unsigned int());
    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& [instanceName,geometryName] : instanceAndGeometryNames)
    {
        if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
        {
            auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
            auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
            for (auto& [meshName, meshData] : geometry.meshes)
            {
                auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                for (auto i = 0; i < desc.recordCount; ++i)
                {
                    auto hitgroupData = HitgroupData();
                    if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                    {
                        hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                    }
                    tracer.pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                    hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                }
                tracer.pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
            }
        }
    }
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(), params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            if (m_CurTracerName == "RIS") {
                params.flags = PARAM_FLAG_NEE | PARAM_FLAG_RIS;

                if (m_EnableGrid) {
                    params.flags |= PARAM_FLAG_USE_GRID;
                    params.mortonTree = m_MortonQuadTree->GetGpuHandle();
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false;
            {
                if (m_EnableGrid) {
                    m_HashBufferCUDA.Update(m_Opx7Context.get(), stream);
                    if (stream) {
                        shouldSync = true;
                    }
                }
            } 
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["RIS"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitDbgTracer() {
    auto tracer = rtlib::test::TracerData();
    tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
    tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
    tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
    tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
    tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
    tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 1;
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
        tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.DBG", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.DBG", moduleOptions);
        tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.DBG", moduleOptions);
    }
    tracer.pipelines["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.DBG", "__raygen__debug");
    tracer.pipelines["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.DBG", "__exception__ep");
    tracer.pipelines["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Debug", "SimpleKernel.DBG", "__miss__debug");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Debug.Triangle", "SimpleKernel.DBG", "__closesthit__debug", "", "", "BuiltIn.Triangle.DBG", "");
    tracer.pipelines["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Debug.Sphere", "SimpleKernel.DBG", "__closesthit__debug_sphere", "", "", "BuiltIn.Sphere.DBG", "");
    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Debug",
        "SimpleKernel.Debug" };

    tracer.pipelines["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    tracer.pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Debug", MissData{ make_float4(m_BgColor,1.0f) });
    tracer.pipelines["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Debug", MissData{});
    tracer.pipelines["Trace"].SetHostExceptionRecordTypeData(unsigned int());
    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
    {
        if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
        {
            auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
            auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
            for (auto& [meshName, meshData] : geometry.meshes)
            {
                auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                for (auto i = 0; i < desc.recordCount; ++i)
                {
                    auto hitgroupData = HitgroupData();
                    if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                    {
                        hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                    }
                    tracer.pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                    hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                }
                tracer.pipelines["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
            }
        }
    }
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {

        params.flags = PARAM_FLAG_NEE;
    }
    tracer.InitParams(m_Opx7Context.get(), params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            if (m_CurTracerName == "DBG") {
                if (m_EnableGrid) {
                    params.flags |= PARAM_FLAG_USE_GRID;
                    params.mortonTree = m_MortonQuadTreeController->GetGpuHandle();
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = stream!=nullptr;
            return shouldSync;
        }
    );
    m_TracerMap["DBG"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitSdTreeDefTracer()
{
    auto tracer = rtlib::test::TracerData();
    {
        //Build
        {
            tracer.pipelines["Build"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Build"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Build"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Build"].linkOptions.maxTraceDepth = 1;

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
            tracer.pipelines["Build"].LoadModule(m_Opx7Context.get(),                      "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get()  ,   "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 1;

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
            tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            tracer.pipelines["Final"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Final"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Final"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Final"].linkOptions.maxTraceDepth = 1;

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
            tracer.pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
    }
    auto stNames = std::vector<std::string>{ "Trace","Build","Final" };
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        tracer.pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    tracer.pipelines["Build"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Build"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();
    
    tracer.pipelines["Final"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{ make_float4(m_BgColor,1.0f) });
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        tracer.pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
        {
            if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                    auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                        }
                        tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                    }
                    tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                }
            }
        }
    }

    tracer.pipelines["Build"].shaderTable->Upload();
    tracer.pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Final"].shaderTable->Upload();
    tracer.pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(),params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            {
                m_SdTreeController->BegTrace(stream);
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecord)
                {
                    return "Build";
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecordAndSample) {
                    return "Trace";
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateSample) {
                    return "Final";
                }
            }
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            {
                params.flags = PARAM_FLAG_NONE;
                if (m_EnableTree) {
                    params.tree = m_SdTreeController->GetGpuSTree();
                    params.flags |= PARAM_FLAG_USE_TREE;
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecord)
                {
                    params.flags |= PARAM_FLAG_NONE;
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecordAndSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                    params.flags |= PARAM_FLAG_FINAL;
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false;
            {
                if (m_EnableTree) {
                    m_SdTreeController->EndTrace(stream);
                    if (stream) {
                        shouldSync = true;
                    }
                }
            }
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["PGDEF"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitSdTreeNeeTracer()
{
    auto tracer = rtlib::test::TracerData();
    {
        //Build
        {
            tracer.pipelines["Build"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Build"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Build"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Build"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            tracer.pipelines["Final"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Final"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Final"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Final"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Trace","Build","Final" };
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        tracer.pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    tracer.pipelines["Build"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Build"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Final"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{ make_float4(m_BgColor,1.0f) });
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        tracer.pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
        {
            if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                    auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                        }
                        tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                    }
                    tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                }
            }
        }
    }

    tracer.pipelines["Build"].shaderTable->Upload();
    tracer.pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Final"].shaderTable->Upload();
    tracer.pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(), params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            {
                m_SdTreeController->BegTrace(stream);
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecord)
                {
                    return "Build";
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecordAndSample) {
                    return "Trace";
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateSample) {
                    return "Final";
                }
            }
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            {
                params.flags = PARAM_FLAG_NONE;
                if (m_EnableTree) {
                    params.tree = m_SdTreeController->GetGpuSTree();
                    params.flags |= PARAM_FLAG_USE_TREE;
                }

                {
                    params.flags |= PARAM_FLAG_NEE;
                }

                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecord)
                {
                    params.flags |= PARAM_FLAG_NONE;
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecordAndSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                    params.flags |= PARAM_FLAG_FINAL;
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false;
            {
                if (m_EnableTree) {
                    m_SdTreeController->EndTrace(stream);
                    if (stream) {
                        shouldSync = true;
                    }
                }
            }
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["PGNEE"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitSdTreeRisTracer()
{
    auto tracer = rtlib::test::TracerData();
    {
        //Build
        {
            tracer.pipelines["Build"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Build"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Build"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Build"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            tracer.pipelines["Final"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Final"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Final"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Final"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            tracer.pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Trace","Build","Final" };
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        tracer.pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    tracer.pipelines["Build"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Build"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Final"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{ make_float4(m_BgColor,1.0f) });
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        tracer.pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
        {
            if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                    auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                        }
                        tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                    }
                    tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                }
            }
        }
    }

    tracer.pipelines["Build"].shaderTable->Upload();
    tracer.pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Final"].shaderTable->Upload();
    tracer.pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(), params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            {
                m_SdTreeController->BegTrace(stream);
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecord)
                {
                    return "Build";
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecordAndSample) {
                    return "Trace";
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateSample) {
                    return "Final";
                }
            }
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            {
                params.flags = PARAM_FLAG_NONE;
                if (m_EnableTree) {
                    params.tree = m_SdTreeController->GetGpuSTree();
                    params.flags |= PARAM_FLAG_USE_TREE;
                }
                {
                    params.flags |= PARAM_FLAG_NEE;
                    params.flags |= PARAM_FLAG_RIS;
                }

                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecord)
                {
                    params.flags |= PARAM_FLAG_NONE;
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateRecordAndSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                }
                if (m_SdTreeController->GetState() == rtlib::test::RTSTreeController::TraceStateSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                    params.flags |= PARAM_FLAG_FINAL;
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false;
            {
                if (m_EnableTree) {
                    m_SdTreeController->EndTrace(stream);
                    if (stream) {
                        shouldSync = true;
                    }
                }
            }
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["PGRIS"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitHashTreeDefTracer()
{
    auto tracer = rtlib::test::TracerData();
    {
        //Locate
        {
            tracer.pipelines["Locate"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Locate"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Locate"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Locate"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Locate"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Locate"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Locate"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Locate"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Locate"].linkOptions.maxTraceDepth = 1;

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
            tracer.pipelines["Locate"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Locate"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Locate"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Build
        {
            tracer.pipelines["Build"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Build"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Build"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Build"].linkOptions.maxTraceDepth = 1;

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
            tracer.pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 1;

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
            tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            tracer.pipelines["Final"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Final"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Final"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Final"].linkOptions.maxTraceDepth = 1;

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
            tracer.pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Locate","Trace","Build","Final" };
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        tracer.pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }

    tracer.pipelines["Locate"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Locate"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Build"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Build"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Final"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{ make_float4(m_BgColor,1.0f) });
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        tracer.pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
        {
            if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                    auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                        }
                        tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                    }
                    tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                }
            }
        }
    }

    tracer.pipelines["Locate"].shaderTable->Upload();
    tracer.pipelines["Locate"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Build"].shaderTable->Upload();
    tracer.pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Final"].shaderTable->Upload();
    tracer.pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(), params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            {
                m_MortonQuadTreeController->BegTrace(stream);
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateLocate)
                {
                    return "Locate";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecord)
                {
                    return "Build";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecordAndSample) {
                    return "Trace";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateSample) {
                    return "Final";
                }
            }
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            {
                if (m_EnableGrid) {
                    params.mortonTree = m_MortonQuadTreeController->GetGpuHandle();
                    params.grid = m_HashBufferCUDA.GetHandle();
                    params.flags |= PARAM_FLAG_USE_GRID;
                    params.mortonTree.level = RTLib::Ext::CUDA::Math::min(params.mortonTree.level, rtlib::test::MortonQTreeWrapper::kMaxTreeLevel);
                }

                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateLocate)
                {
                    params.flags |= PARAM_FLAG_LOCATE;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecord)
                {
                    params.flags |= PARAM_FLAG_NONE;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecordAndSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                    params.flags |= PARAM_FLAG_FINAL;
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false;
            {
                m_MortonQuadTreeController->EndTrace(stream);
                if (m_EnableGrid) {
                    if (m_MortonQuadTreeController->GetSamplePerTmp() == 0)
                    {
                        m_HashBufferCUDA.Update(m_Opx7Context.get(), stream);
                    }
                }
                if (stream) {
                    shouldSync = true;
                }
            }
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["HTDEF"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitHashTreeNeeTracer()
{
    auto tracer = rtlib::test::TracerData();
    {
        //Locate
        {
            tracer.pipelines["Locate"].compileOptions.usesMotionBlur        = false;
            tracer.pipelines["Locate"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Locate"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Locate"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Locate"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Locate"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Locate"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Locate"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Locate"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Locate"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Locate"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Locate"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Build
        {
            tracer.pipelines["Build"].compileOptions.usesMotionBlur            = false;
            tracer.pipelines["Build"].compileOptions.traversableGraphFlags     = 0;
            tracer.pipelines["Build"].compileOptions.numAttributeValues        = 3;
            tracer.pipelines["Build"].compileOptions.numPayloadValues          = 8;
            tracer.pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Build"].compileOptions.usesPrimitiveTypeFlags    = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Build"].compileOptions.exceptionFlags            = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Build"].linkOptions.debugLevel                   = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Build"].linkOptions.maxTraceDepth                = 2;

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
            tracer.pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            tracer.pipelines["Final"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Final"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Final"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Final"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Locate","Trace","Build","Final"};
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        tracer.pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    tracer.pipelines["Locate"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Locate"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Build"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Build"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Final"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{ make_float4(m_BgColor,1.0f) });
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        tracer.pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());


        for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
        {
            if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                    auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                        }
                        tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                    }
                    tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                }
            }
        }
    }

    tracer.pipelines["Locate"].shaderTable->Upload();
    tracer.pipelines["Locate"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    tracer.pipelines["Build" ].shaderTable->Upload();
    tracer.pipelines["Build" ].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    tracer.pipelines["Trace" ].shaderTable->Upload();
    tracer.pipelines["Trace" ].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    tracer.pipelines["Final" ].shaderTable->Upload();
    tracer.pipelines["Final" ].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(), params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            {
                m_MortonQuadTreeController->BegTrace(stream);
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateLocate)
                {
                    return "Locate";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecord)
                {
                    return "Build";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecordAndSample) {
                    return "Trace";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateSample) {
                    return "Final";
                }
            }
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            
            {
                if (m_EnableGrid) {
                    params.mortonTree = m_MortonQuadTreeController->GetGpuHandle();
                    params.grid = m_HashBufferCUDA.GetHandle();
                    params.flags |= PARAM_FLAG_USE_GRID;
                    params.mortonTree.level = RTLib::Ext::CUDA::Math::min(params.mortonTree.level, rtlib::test::MortonQTreeWrapper::kMaxTreeLevel);
                }
                {
                    params.flags |= PARAM_FLAG_NEE;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateLocate)
                {
                    params.flags |= PARAM_FLAG_LOCATE;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecord)
                {
                    params.flags |= PARAM_FLAG_NONE;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecordAndSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                    params.flags |= PARAM_FLAG_FINAL;
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false; {
                m_MortonQuadTreeController->EndTrace(stream);
                if (m_EnableGrid) {
                    if (m_MortonQuadTreeController->GetSamplePerTmp() == 0)
                    {
                        m_HashBufferCUDA.Update(m_Opx7Context.get(), stream);
                    }
                }
                if (stream) {
                    shouldSync = true;
                }
            }
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["HTNEE"] = std::move(tracer);
}

void RTLibExtOPX7TestApplication::InitHashTreeRisTracer()
{
    auto tracer = rtlib::test::TracerData();
    {
        //Locate
        {
            tracer.pipelines["Locate"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Locate"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Locate"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Locate"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Locate"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Locate"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Locate"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Locate"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Locate"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Locate"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Locate"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Locate"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Build
        {
            tracer.pipelines["Build"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Build"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Build"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Build"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Build"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Build"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Build"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Build"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Build"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Build"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Build"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Build"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Trace
        {
            tracer.pipelines["Trace"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Trace"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Trace"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Trace"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Trace"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Trace"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Trace"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }
        //Final
        {
            tracer.pipelines["Final"].compileOptions.usesMotionBlur = false;
            tracer.pipelines["Final"].compileOptions.traversableGraphFlags = 0;
            tracer.pipelines["Final"].compileOptions.numAttributeValues = 3;
            tracer.pipelines["Final"].compileOptions.numPayloadValues = 8;
            tracer.pipelines["Final"].compileOptions.launchParamsVariableNames = "params";
            tracer.pipelines["Final"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
            tracer.pipelines["Final"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
            tracer.pipelines["Final"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
            tracer.pipelines["Final"].linkOptions.maxTraceDepth = 2;

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
            tracer.pipelines["Final"].LoadModule(m_Opx7Context.get(), "SimpleKernel.PGDEF", moduleOptions, m_PtxStringMap.at("SimpleGuide2.ptx"));
            tracer.pipelines["Final"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.PGDEF", moduleOptions);
            tracer.pipelines["Final"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.PGDEF", moduleOptions);
        }

    }
    auto stNames = std::vector<std::string>{ "Locate","Trace","Build","Final" };
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__raygen__default");
        tracer.pipelines[stName].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.PGDEF", "__exception__ep");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.PGDEF", "__miss__radiance");
        tracer.pipelines[stName].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.PGDEF", "__miss__occluded");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Triangle", "SimpleKernel.PGDEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Triangle", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance.Sphere", "SimpleKernel.PGDEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.PGDEF", "");
        tracer.pipelines[stName].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded.Sphere", "SimpleKernel.PGDEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.PGDEF", "");
    }
    tracer.pipelines["Locate"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Locate"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Build"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Build"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Trace"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Trace"].shaderTable = this->NewShaderTable();

    tracer.pipelines["Final"].InitPipeline(m_Opx7Context.get());
    tracer.pipelines["Final"].shaderTable = this->NewShaderTable();

    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded"
    };
    auto instanceAndGeometryNames = rtlib::test::EnumerateGeometriesFromGASInstances(m_ShaderTableLayout.get(), m_SceneData.world);
    for (auto& stName : stNames) {
        tracer.pipelines[stName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{ make_float4(m_BgColor,1.0f) });
        tracer.pipelines[stName].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
        tracer.pipelines[stName].SetHostExceptionRecordTypeData(unsigned int());

        for (auto& [instanceName, geometryName] : instanceAndGeometryNames)
        {
            if (m_SceneData.world.geometryObjModels.count(geometryName) > 0)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                    auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                    for (auto i = 0; i < desc.recordCount; ++i)
                    {
                        auto hitgroupData = HitgroupData();
                        if ((i % desc.recordStride) == RAY_TYPE_RADIANCE)
                        {
                            hitgroupData = rtlib::test::GetHitgroupFromObjMesh(meshData.materials[i / desc.recordStride], mesh, m_TextureMap);
                        }
                        tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        hitgroupData = rtlib::test::GetHitgroupFromSphere(sphereRes->materials[i / desc.recordStride], sphereRes, m_TextureMap);
                    }
                    tracer.pipelines[stName].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                }
            }
        }
    }

    tracer.pipelines["Locate"].shaderTable->Upload();
    tracer.pipelines["Locate"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Build"].shaderTable->Upload();
    tracer.pipelines["Build"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Trace"].shaderTable->Upload();
    tracer.pipelines["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    tracer.pipelines["Final"].shaderTable->Upload();
    tracer.pipelines["Final"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());

    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    tracer.InitParams(m_Opx7Context.get(), params);
    tracer.beginTrace = std::function<std::string(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream) {
            {
                m_MortonQuadTreeController->BegTrace(stream);
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateLocate)
                {
                    return "Locate";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecord)
                {
                    return "Build";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecordAndSample) {
                    return "Trace";
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateSample) {
                    return "Final";
                }
            }
            return "Trace";
        }
    );
    tracer.getParams = std::function<Params(RTLib::Ext::CUDA::CUDABuffer*)>(
        [this](RTLib::Ext::CUDA::CUDABuffer* frameBuffer) {
            auto params = Params();
            params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
            params.seedBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
            params.frameBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<uchar4>(frameBuffer);
            params.width = m_SceneData.config.width;
            params.height = m_SceneData.config.height;
            params.maxDepth = m_SceneData.config.maxDepth;
            params.samplesForAccum = m_SamplesForAccum;
            params.samplesForLaunch = m_SceneData.config.samples;
            params.debugFrameType = m_DebugFrameType;
            params.gasHandle = m_InstanceASMap["Root"].handle;
            params.lights.count = m_lightBuffer.cpuHandle.size();
            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
            params.grid = m_HashBufferCUDA.GetHandle();
            params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates", 32);
            //std::cout << params.maxDepth << std::endl;
            if (m_EnableGrid) {
                params.diffuseGridBuffer = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float4>(m_DiffuseBufferCUDA.get());
            }
            {
                if (m_EnableGrid) {
                    params.mortonTree = m_MortonQuadTreeController->GetGpuHandle();
                    params.grid = m_HashBufferCUDA.GetHandle();
                    params.flags |= PARAM_FLAG_USE_GRID;
                    params.mortonTree.level = RTLib::Ext::CUDA::Math::min(params.mortonTree.level, rtlib::test::MortonQTreeWrapper::kMaxTreeLevel);
                }

                {
                    params.flags |= PARAM_FLAG_NEE;
                    params.flags |= PARAM_FLAG_RIS;
                }

                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateLocate)
                {
                    params.flags |= PARAM_FLAG_LOCATE;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecord)
                {
                    params.flags |= PARAM_FLAG_NONE;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateRecordAndSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                }
                if (m_MortonQuadTreeController->GetState() == rtlib::test::RTMortonQuadTreeController::TraceStateSample)
                {
                    params.flags |= PARAM_FLAG_BUILD;
                    params.flags |= PARAM_FLAG_FINAL;
                }
            }
            return params;
        }
    );
    tracer.endTrace = std::function<bool(RTLib::Ext::CUDA::CUDAStream*)>(
        [this](RTLib::Ext::CUDA::CUDAStream* stream)->bool {
            bool shouldSync = false;
            {
                m_MortonQuadTreeController->EndTrace(stream);
                if (m_EnableGrid) {
                    if (m_MortonQuadTreeController->GetSamplePerTmp() == 0)
                    {
                        m_HashBufferCUDA.Update(m_Opx7Context.get(), stream);
                    }
                }
                if (stream) {
                    shouldSync = true;
                }
            }
            {
                m_SamplesForAccum += m_SceneData.config.samples;
            }
            return shouldSync;
        }
    );
    m_TracerMap["HTRIS"] = std::move(tracer);
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

 void RTLibExtOPX7TestApplication::SetupPipeline()
 {
     auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
     desc.sizeInBytes = m_FrameBufferCUDA->GetSizeInBytes();
     auto frameBufferTmp = m_Opx7Context->CreateBuffer(desc);
     m_TracerMap[m_CurTracerName].TracePipeline(m_Opx7Context.get(), nullptr, frameBufferTmp, m_SceneData.config.width, m_SceneData.config.height);
     m_SamplesForAccum = 0;
     m_TimesForIterations = std::vector<unsigned long long>{ 0 };
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
            m_TracerMap[m_CurTracerName].GetCurPipeline().SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
            m_TracerMap[m_CurTracerName].GetCurPipeline().shaderTable->UploadRaygenRecord();
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
        (void)m_TracerMap[m_CurTracerName].TracePipeline(m_Opx7Context.get(), stream, frameBufferCUDA, m_SceneData.config.width,m_SceneData.config.height);
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
        if (this->m_TracerMap[m_CurTracerName].TracePipeline(m_Opx7Context.get(), stream, m_FrameBufferCUDA.get(),m_SceneData.config.width, m_SceneData.config.height)) {
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
        if ((m_CurTracerName == "HTDEF") ||
            (m_CurTracerName == "HTRIS")) {
            m_TimesForIterations.back() += m_TimesForFrame;
            if (m_MortonQuadTreeController->GetIteration() != m_TimesForIterations.size()) {
                m_TimesForIterations.push_back(0);
            }
        }
        if ((m_CurTracerName == "PGDEF") ||
            (m_CurTracerName == "PGRIS")) {
            m_TimesForIterations.back() += m_TimesForFrame;
            if (m_SdTreeController->GetIteration() != m_TimesForIterations.size()) {
                m_TimesForIterations.push_back(0);
            }
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
     if ((m_CurTracerName == "PGDEF") ||
         (m_CurTracerName == "PGRIS") ||
         (m_CurTracerName == "HTDEF") ||
         (m_CurTracerName == "HTRIS")) {
         configData.custom.SetUInt32("NumIterations", m_TimesForIterations.size());
         unsigned int i = 0;
         for (auto& time : m_TimesForIterations) {
             configData.custom.SetFloat1("TimesForItrerations[" + std::to_string(i) + "]", time/(1000.0f*1000));
             ++i;
         }
     }
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
         RTLib::Core::SaveExrImage(configData.exrFilePath.c_str(), m_SceneData.config.width, m_SceneData.config.height, hdr_image_data);
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
         RTLib::Core::SavePngImage(configData.pngFilePath.c_str(), m_SceneData.config.width, m_SceneData.config.height, png_image_data);
     }
 }