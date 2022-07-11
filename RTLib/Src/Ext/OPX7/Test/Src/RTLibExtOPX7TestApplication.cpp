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

void RTLibExtOPX7TestApplication::LoadScene()
{
    m_SceneData = rtlib::test::LoadScene(m_ScenePath);
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
                            bool hasLight = false;
                            bool useNEE = false;
                            if (mesh->GetUniqueResource()->variables.HasBool("hasLight"))
                            {
                                hasLight = mesh->GetUniqueResource()->variables.GetBool("hasLight");
                            }
                            if (hasLight)
                            {
                                auto meshLight = MeshLight();
                                meshLight.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                meshLight.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                meshLight.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                meshLight.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                meshLight.indCount = mesh->GetUniqueResource()->triIndBuffer.size();
                                meshLight.emission = meshData.materials.front().GetFloat3As<float3>("emitCol");
                                auto emitTexStr = meshData.materials.front().GetString("emitTex");
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
    m_HashBufferCUDA.aabbMin = make_float3(m_WorldAabbMin[0], m_WorldAabbMin[1], m_WorldAabbMin[2]);
    m_HashBufferCUDA.aabbMax = make_float3(m_WorldAabbMax[0], m_WorldAabbMax[1], m_WorldAabbMax[2]);
    m_HashBufferCUDA.Alloc(make_uint3(1024, 1024, 1024), 1024 * 1024);
    m_HashBufferCUDA.Upload(m_Opx7Context.get());
}

 void RTLibExtOPX7TestApplication::FreeGrids()
{
    m_HashBufferCUDA.dataGpuHandle->Destroy();
    m_HashBufferCUDA.checkSumGpuHandle->Destroy();
}

 void RTLibExtOPX7TestApplication::InitSdTree()
 {
     m_SdTree = std::make_unique<rtlib::test::RTSTreeWrapper>(m_Opx7Context.get(),
         make_float3(m_WorldAabbMin[0], m_WorldAabbMin[1], m_WorldAabbMin[2]),
         make_float3(m_WorldAabbMax[0], m_WorldAabbMax[1], m_WorldAabbMax[2])
     );
     m_SdTree->Upload();
 }

 void RTLibExtOPX7TestApplication::FreeSdTree()
 {
     if (m_SdTree) {
         m_SdTree->Destroy();
         m_SdTree.reset();
     }
 }

 void RTLibExtOPX7TestApplication::InitPtxString()
{
    m_PtxStringMap["SimpleTrace.ptx"] = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleTrace.optixir");
    m_PtxStringMap["SimpleGuide.ptx"] = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleGuide.optixir");
}

void RTLibExtOPX7TestApplication::InitDefTracer()
{
    m_TracerMap["DEF"].pipelineName = "Trace";
    m_TracerMap["DEF"].pipelineDatas["Trace"].compileOptions.usesMotionBlur = false;
    m_TracerMap["DEF"].pipelineDatas["Trace"].compileOptions.traversableGraphFlags = 0;
    m_TracerMap["DEF"].pipelineDatas["Trace"].compileOptions.numAttributeValues = 3;
    m_TracerMap["DEF"].pipelineDatas["Trace"].compileOptions.numPayloadValues = 8;
    m_TracerMap["DEF"].pipelineDatas["Trace"].compileOptions.launchParamsVariableNames = "params";
    m_TracerMap["DEF"].pipelineDatas["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle| RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    m_TracerMap["DEF"].pipelineDatas["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_TracerMap["DEF"].pipelineDatas["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_TracerMap["DEF"].pipelineDatas["Trace"].linkOptions.maxTraceDepth = 1;
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
        moduleOptions.boundValueEntries.front().annotation    = "flags(PARAM_FLAG_NONE)";

        m_TracerMap["DEF"].pipelineDatas["Trace"].LoadModule(m_Opx7Context.get(), "SimpleTrace.DEF", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        m_TracerMap["DEF"].pipelineDatas["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.DEF", moduleOptions);
        m_TracerMap["DEF"].pipelineDatas["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(),   "BuiltIn.Sphere.DEF", moduleOptions);
    }
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleTrace.DEF", "__raygen__default");
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleTrace.DEF", "__exception__ep");
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleTrace.Radiance", "SimpleTrace.DEF", "__miss__radiance");
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleTrace.Occluded", "SimpleTrace.DEF", "__miss__occluded");
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Radiance.Triangle", "SimpleTrace.DEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.DEF", "");
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Occluded.Triangle", "SimpleTrace.DEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.DEF", "");
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Radiance.Sphere"  , "SimpleTrace.DEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.DEF", "");
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Occluded.Sphere"  , "SimpleTrace.DEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.DEF", "");
    m_TracerMap["DEF"].pipelineDatas["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["DEF"].pipelineDatas["Trace"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
    {

        auto shaderTableDesc = RTLib::Ext::OPX7::OPX7ShaderTableCreateDesc();
        {

            shaderTableDesc.raygenRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
            shaderTableDesc.missRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
            shaderTableDesc.missRecordCount = m_ShaderTableLayout->GetRecordStride();
            shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
            shaderTableDesc.hitgroupRecordCount = m_ShaderTableLayout->GetRecordCount();
            shaderTableDesc.exceptionRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<unsigned int>);
        }
        m_TracerMap["DEF"].pipelineDatas["Trace"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleTrace.Radiance",
        "SimpleTrace.Occluded" 
    };
    auto raygenRecord = m_TracerMap["DEF"].pipelineDatas["Trace"].programGroupRG->GetRecord<RayGenData>();
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleTrace.Radiance", MissData{});
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleTrace.Occluded", MissData{});
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetHostExceptionRecordTypeData(unsigned int());

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
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
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
                            m_TracerMap["DEF"].pipelineDatas["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride]+".Triangle", hitgroupData);
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
                        m_TracerMap["DEF"].pipelineDatas["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                    }
                }
            }
        }
    }
    m_TracerMap["DEF"].pipelineDatas["Trace"].shaderTable->Upload();
    m_TracerMap["DEF"].pipelineDatas["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["DEF"].pipelineDatas["Trace"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
}

void RTLibExtOPX7TestApplication::InitNeeTracer()
{
    m_TracerMap["NEE"].pipelineName = "Trace";
    m_TracerMap["NEE"].pipelineDatas["Trace"].compileOptions.usesMotionBlur = false;
    m_TracerMap["NEE"].pipelineDatas["Trace"].compileOptions.traversableGraphFlags = 0;
    m_TracerMap["NEE"].pipelineDatas["Trace"].compileOptions.numAttributeValues = 3;
    m_TracerMap["NEE"].pipelineDatas["Trace"].compileOptions.numPayloadValues = 8;
    m_TracerMap["NEE"].pipelineDatas["Trace"].compileOptions.launchParamsVariableNames = "params";
    m_TracerMap["NEE"].pipelineDatas["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    m_TracerMap["NEE"].pipelineDatas["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_TracerMap["NEE"].pipelineDatas["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_TracerMap["NEE"].pipelineDatas["Trace"].linkOptions.maxTraceDepth = 2;
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
        if (m_EnableGrid) {
            moduleOptions.boundValueEntries.front().annotation = "flags(PARAM_FLAG_NEE|PARAM_FLAG_GRID)";
        }
        else {
            moduleOptions.boundValueEntries.front().annotation = "flags(PARAM_FLAG_NEE)";
        }
        m_TracerMap["NEE"].pipelineDatas["Trace"].LoadModule(m_Opx7Context.get(), "SimpleTrace.NEE", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        m_TracerMap["NEE"].pipelineDatas["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.NEE", moduleOptions);
        m_TracerMap["NEE"].pipelineDatas["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.NEE", moduleOptions);
    }
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleTrace.NEE", "__raygen__default");
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleTrace.NEE", "__exception__ep");
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleTrace.Radiance", "SimpleTrace.NEE", "__miss__radiance");
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleTrace.Occluded", "SimpleTrace.NEE", "__miss__occluded");
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Radiance.Triangle", "SimpleTrace.NEE", "__closesthit__radiance", "", "", "BuiltIn.Triangle.NEE", "");
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Occluded.Triangle", "SimpleTrace.NEE", "__closesthit__occluded", "", "", "BuiltIn.Triangle.NEE", "");
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Radiance.Sphere", "SimpleTrace.NEE", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.NEE", "");
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Occluded.Sphere", "SimpleTrace.NEE", "__closesthit__occluded", "", "", "BuiltIn.Sphere.NEE", "");
    m_TracerMap["NEE"].pipelineDatas["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["NEE"].pipelineDatas["Trace"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
    {

        auto shaderTableDesc = RTLib::Ext::OPX7::OPX7ShaderTableCreateDesc();
        {

            shaderTableDesc.raygenRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
            shaderTableDesc.missRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
            shaderTableDesc.missRecordCount = m_ShaderTableLayout->GetRecordStride();
            shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
            shaderTableDesc.hitgroupRecordCount = m_ShaderTableLayout->GetRecordCount();
            shaderTableDesc.exceptionRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<unsigned int>);
        }
        m_TracerMap["NEE"].pipelineDatas["Trace"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleTrace.Radiance",
        "SimpleTrace.Occluded" };
    auto raygenRecord = m_TracerMap["NEE"].pipelineDatas["Trace"].programGroupRG->GetRecord<RayGenData>();
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleTrace.Radiance", MissData{});
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleTrace.Occluded", MissData{});
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetHostExceptionRecordTypeData(unsigned int());

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
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
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
                            m_TracerMap["NEE"].pipelineDatas["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        m_TracerMap["NEE"].pipelineDatas["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                    }
                }
            }
        }
    }
    m_TracerMap["NEE"].pipelineDatas["Trace"].shaderTable->Upload();
    m_TracerMap["NEE"].pipelineDatas["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["NEE"].pipelineDatas["Trace"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
}

void RTLibExtOPX7TestApplication::InitRisTracer()
{
    m_TracerMap["RIS"].pipelineName = "Trace";
    m_TracerMap["RIS"].pipelineDatas["Trace"].compileOptions.usesMotionBlur = false;
    m_TracerMap["RIS"].pipelineDatas["Trace"].compileOptions.traversableGraphFlags = 0;
    m_TracerMap["RIS"].pipelineDatas["Trace"].compileOptions.numAttributeValues = 3;
    m_TracerMap["RIS"].pipelineDatas["Trace"].compileOptions.numPayloadValues = 8;
    m_TracerMap["RIS"].pipelineDatas["Trace"].compileOptions.launchParamsVariableNames = "params";
    m_TracerMap["RIS"].pipelineDatas["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    m_TracerMap["RIS"].pipelineDatas["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_TracerMap["RIS"].pipelineDatas["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_TracerMap["RIS"].pipelineDatas["Trace"].linkOptions.maxTraceDepth = 2;
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
        if (m_EnableGrid) {

            moduleOptions.boundValueEntries.front().annotation = "flags(PARAM_FLAG_RIS|PARAM_FLAG_NEE|PARAM_FLAG_GRID)";
        }
        else {
            moduleOptions.boundValueEntries.front().annotation = "flags(PARAM_FLAG_RIS|PARAM_FLAG_NEE)";
        }
        m_TracerMap["RIS"].pipelineDatas["Trace"].LoadModule(m_Opx7Context.get(), "SimpleTrace.RIS", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        m_TracerMap["RIS"].pipelineDatas["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.RIS", moduleOptions);
        m_TracerMap["RIS"].pipelineDatas["Trace"].LoadBuiltInISSphereModule(m_Opx7Context.get(), "BuiltIn.Sphere.RIS", moduleOptions);
    }
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleTrace.RIS", "__raygen__default");
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleTrace.RIS", "__exception__ep");
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleTrace.Radiance", "SimpleTrace.RIS", "__miss__radiance");
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleTrace.Occluded", "SimpleTrace.RIS", "__miss__occluded");
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Radiance.Triangle", "SimpleTrace.RIS", "__closesthit__radiance", "", "", "BuiltIn.Triangle.RIS", "");
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Occluded.Triangle", "SimpleTrace.RIS", "__closesthit__occluded", "", "", "BuiltIn.Triangle.RIS", "");
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Radiance.Sphere", "SimpleTrace.RIS", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.RIS", "");
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Occluded.Sphere", "SimpleTrace.RIS", "__closesthit__occluded", "", "", "BuiltIn.Sphere.RIS", "");
    m_TracerMap["RIS"].pipelineDatas["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["RIS"].pipelineDatas["Trace"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
    {

        auto shaderTableDesc = RTLib::Ext::OPX7::OPX7ShaderTableCreateDesc();
        {

            shaderTableDesc.raygenRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
            shaderTableDesc.missRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
            shaderTableDesc.missRecordCount = m_ShaderTableLayout->GetRecordStride();
            shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
            shaderTableDesc.hitgroupRecordCount = m_ShaderTableLayout->GetRecordCount();
            shaderTableDesc.exceptionRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<unsigned int>);
        }
        m_TracerMap["RIS"].pipelineDatas["Trace"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleTrace.Radiance",
        "SimpleTrace.Occluded" };
    auto raygenRecord = m_TracerMap["RIS"].pipelineDatas["Trace"].programGroupRG->GetRecord<RayGenData>();
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleTrace.Radiance", MissData{});
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleTrace.Occluded", MissData{});
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetHostExceptionRecordTypeData(unsigned int());

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
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
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
                            m_TracerMap["RIS"].pipelineDatas["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        m_TracerMap["RIS"].pipelineDatas["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                    }
                }
            }
        }
    }
    m_TracerMap["RIS"].pipelineDatas["Trace"].shaderTable->Upload();
    m_TracerMap["RIS"].pipelineDatas["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_TracerMap["RIS"].pipelineDatas["Trace"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
}

void RTLibExtOPX7TestApplication::InitDefGuideTracer()
{
    auto compileOptions = RTLib::Ext::OPX7::OPX7PipelineCompileOptions();
    compileOptions.usesMotionBlur = false;
    compileOptions.traversableGraphFlags = 0;
    compileOptions.numAttributeValues = 3;
    compileOptions.numPayloadValues = 8;
    compileOptions.launchParamsVariableNames = "params";
    compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;

    auto linkOptions = RTLib::Ext::OPX7::OPX7PipelineLinkOptions();
    linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    linkOptions.maxTraceDepth = 1;

    auto pipelineNames = std::vector<std::string>{
        "BUILD",
        "TRACE",
        "FINAL"
    };
    auto annotations = std::vector<std::string>{
        "PARAM_FLAG_NONE",
        "PARAM_FLAG_BUILD",
        "PARAM_FLAG_BUILD|PARAM_FLAG_FINAL"
    };
    {
        for (auto& annotation : annotations)
        {
            annotation = "flag(" + annotation;
            if (m_EnableTree) {
                annotation = annotation + "|PARAM_FLAG_USE_TREE";
            }
            if (m_EnableGrid) {
                annotation = annotation + "|PARAM_FLAG_USE_GRID";
            }
            annotation += ")";

        }
    }
    auto paramFlags = std::vector<unsigned int>{
        PARAM_FLAG_NONE ,
        PARAM_FLAG_BUILD,
        PARAM_FLAG_BUILD|PARAM_FLAG_FINAL,
    };
    {
        for (auto& flag : paramFlags) {
            if (m_EnableTree) {
                flag |= PARAM_FLAG_USE_TREE;
            }

            if (m_EnableGrid) {
                flag |= PARAM_FLAG_USE_GRID;
            }
        }
        m_TracerMap["PGDEF"].pipelineName = "BUILD";

        for (auto m = 0; m < pipelineNames.size(); ++m) {
            auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};

            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].compileOptions = compileOptions;
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].linkOptions    = linkOptions;
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
            moduleOptions.boundValueEntries.front().boundValuePtr              = &paramFlags[m];
            moduleOptions.boundValueEntries.front().sizeInBytes                = sizeof(paramFlags[m]);
            moduleOptions.boundValueEntries.front().annotation                 = annotations[m].c_str();

            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].LoadModule(m_Opx7Context.get(), "SimpleGuide.DEF", moduleOptions, m_PtxStringMap.at("SimpleGuide.ptx"));
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.DEF", moduleOptions);
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].LoadBuiltInISSphereModule(  m_Opx7Context.get(), "BuiltIn.Sphere.DEF"  , moduleOptions);
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetProgramGroupRG(m_Opx7Context.get(), "SimpleGuide.DEF", "__raygen__default");
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetProgramGroupEP(m_Opx7Context.get(), "SimpleGuide.DEF", "__exception__ep");
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetProgramGroupMS(m_Opx7Context.get(), "SimpleGuide.Radiance", "SimpleGuide.DEF", "__miss__radiance");
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetProgramGroupMS(m_Opx7Context.get(), "SimpleGuide.Occluded", "SimpleGuide.DEF", "__miss__occluded");
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetProgramGroupHG(m_Opx7Context.get(), "SimpleGuide.Radiance.Triangle", "SimpleGuide.DEF", "__closesthit__radiance", "", "", "BuiltIn.Triangle.DEF", "");
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetProgramGroupHG(m_Opx7Context.get(), "SimpleGuide.Occluded.Triangle", "SimpleGuide.DEF", "__closesthit__occluded", "", "", "BuiltIn.Triangle.DEF", "");
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetProgramGroupHG(m_Opx7Context.get(), "SimpleGuide.Radiance.Sphere", "SimpleGuide.DEF", "__closesthit__radiance_sphere", "", "", "BuiltIn.Sphere.DEF", "");
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetProgramGroupHG(m_Opx7Context.get(), "SimpleGuide.Occluded.Sphere", "SimpleGuide.DEF", "__closesthit__occluded", "", "", "BuiltIn.Sphere.DEF", "");
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].InitPipeline(m_Opx7Context.get());
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
            {

                auto shaderTableDesc = RTLib::Ext::OPX7::OPX7ShaderTableCreateDesc();
                {

                    shaderTableDesc.raygenRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
                    shaderTableDesc.missRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
                    shaderTableDesc.missRecordCount = m_ShaderTableLayout->GetRecordStride();
                    shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
                    shaderTableDesc.hitgroupRecordCount = m_ShaderTableLayout->GetRecordCount();
                    shaderTableDesc.exceptionRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<unsigned int>);
                }
                m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
            }
            auto programGroupHGNames = std::vector<std::string>{
                "SimpleGuide.Radiance",
                "SimpleGuide.Occluded" 
            };
            auto raygenRecord = m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].programGroupRG->GetRecord<RayGenData>();
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleGuide.Radiance", MissData{});
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleGuide.Occluded", MissData{});
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetHostExceptionRecordTypeData(unsigned int());

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
                                        hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                        hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                        hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
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
                                    m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                                m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Sphere", hitgroupData);
                            }
                        }
                    }
                }
            }
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].shaderTable->Upload();
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
            auto params = Params();
            {
                params.flags = PARAM_FLAG_NONE;
            }
            m_TracerMap["PGDEF"].pipelineDatas[pipelineNames[m]].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
        }
    }
}

void RTLibExtOPX7TestApplication::InitDbgTracer() {
    m_TracerMap["DBG"].pipelineName = "Trace";
    m_TracerMap["DBG"].pipelineDatas["Trace"].compileOptions.usesMotionBlur = false;
    m_TracerMap["DBG"].pipelineDatas["Trace"].compileOptions.traversableGraphFlags = 0;
    m_TracerMap["DBG"].pipelineDatas["Trace"].compileOptions.numAttributeValues = 3;
    m_TracerMap["DBG"].pipelineDatas["Trace"].compileOptions.numPayloadValues = 8;
    m_TracerMap["DBG"].pipelineDatas["Trace"].compileOptions.launchParamsVariableNames = "params";
    m_TracerMap["DBG"].pipelineDatas["Trace"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle | RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsSphere;
    m_TracerMap["DBG"].pipelineDatas["Trace"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_TracerMap["DBG"].pipelineDatas["Trace"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_TracerMap["DBG"].pipelineDatas["Trace"].linkOptions.maxTraceDepth = 1;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_NEE;
        if (m_EnableGrid) {
            flags |= PARAM_FLAG_USE_GRID;
        }
#ifdef NDEBUG
        moduleOptions.optLevel   = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eLevel3;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eNone;
#else
        moduleOptions.optLevel   = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eModerate;
#endif
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        m_TracerMap["DBG"].pipelineDatas["Trace"].LoadModule(m_Opx7Context.get(), "SimpleTrace.DBG", moduleOptions, m_PtxStringMap.at("SimpleTrace.ptx"));
        m_TracerMap["DBG"].pipelineDatas["Trace"].LoadBuiltInISTriangleModule(m_Opx7Context.get(), "BuiltIn.Triangle.DBG", moduleOptions);
        m_TracerMap["DBG"].pipelineDatas["Trace"].LoadBuiltInISSphereModule(  m_Opx7Context.get(), "BuiltIn.Sphere.DBG", moduleOptions);
    }
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleTrace.DBG", "__raygen__debug");
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleTrace.DBG", "__exception__ep");
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleTrace.Debug", "SimpleTrace.DBG", "__miss__debug");
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Debug.Triangle", "SimpleTrace.DBG", "__closesthit__debug", "", "", "BuiltIn.Triangle.DBG", "");
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleTrace.Debug.Sphere"  , "SimpleTrace.DBG", "__closesthit__debug_sphere", "", "", "BuiltIn.Sphere.DBG"  , "");
    m_TracerMap["DBG"].pipelineDatas["Trace"].InitPipeline(m_Opx7Context.get());
    m_TracerMap["DBG"].pipelineDatas["Trace"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
    {

        auto shaderTableDesc = RTLib::Ext::OPX7::OPX7ShaderTableCreateDesc();
        {

            shaderTableDesc.raygenRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
            shaderTableDesc.missRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
            shaderTableDesc.missRecordCount = m_ShaderTableLayout->GetRecordStride();
            shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
            shaderTableDesc.hitgroupRecordCount = m_ShaderTableLayout->GetRecordCount();
            shaderTableDesc.exceptionRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<unsigned int>);
        }
        m_TracerMap["DBG"].pipelineDatas["Trace"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleTrace.Debug",
        "SimpleTrace.Debug" };
    auto raygenRecord = m_TracerMap["DBG"].pipelineDatas["Trace"].programGroupRG->GetRecord<RayGenData>();
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleTrace.Debug", MissData{});
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleTrace.Debug", MissData{});
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetHostExceptionRecordTypeData(unsigned int());
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
                                hitgroupData.type = rtlib::test::SpecifyMaterialType(material);
                                hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
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
                            m_TracerMap["DBG"].pipelineDatas["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride] + ".Triangle", hitgroupData);
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
                        m_TracerMap["DBG"].pipelineDatas["Trace"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride]+".Sphere", hitgroupData);
                    }
                }
            }
        }
    }
    m_TracerMap["DBG"].pipelineDatas["Trace"].shaderTable->Upload();
    m_TracerMap["DBG"].pipelineDatas["Trace"].SetPipelineStackSize(m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {

        params.flags = PARAM_FLAG_NEE;
    }
    m_TracerMap["DBG"].pipelineDatas["Trace"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
}

 void RTLibExtOPX7TestApplication::FreeTracers()
{
    for (auto& [name, tracer] : m_TracerMap)
    {
        tracer.Free();
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

 void RTLibExtOPX7TestApplication::TracePipeline(RTLib::Ext::CUDA::CUDAStream* stream, RTLib::Ext::CUDA::CUDABuffer* frameBuffer)
{
    if (m_CurTracerName == "PGDEF") {
        TraceBegPgDefPipeline();
    }
    auto params = Params();
    {
        params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
        params.seedBuffer = reinterpret_cast<unsigned int*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_SeedBufferCUDA.get()));
        params.frameBuffer = reinterpret_cast<uchar4*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(frameBuffer));
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

        if (m_CurTracerName == "DEF")
        {
            params.flags = PARAM_FLAG_NONE;
        }
        if (m_CurTracerName == "NEE")
        {
            params.flags = PARAM_FLAG_NEE;
        }
        if (m_CurTracerName == "RIS") {
            params.flags = PARAM_FLAG_NEE| PARAM_FLAG_RIS ;
        }
        
        if (m_CurTracerName == "PGDEF") {
            params.flags      = PARAM_FLAG_NONE;
            if (m_EnableTree) {

                auto curIteration      = m_Variables.GetUInt32("CurIteration");
                auto iterationForBuilt = m_Variables.GetUInt32("IterationForBuilt");
                auto sampleForPass     = m_Variables.GetUInt32("SampleForPass"  );
                auto sampleForRemain   = m_Variables.GetUInt32("SampleForRemain");

                params.tree   = m_SdTree->GetGpuHandle();
                params.flags |= PARAM_FLAG_USE_TREE;

                if (curIteration > iterationForBuilt) {
                    if (sampleForPass >= sampleForRemain) {
                        std::cout << "FINAL" << std::endl;
                        m_TracerMap[m_CurTracerName].pipelineName = "FINAL";
                        params.flags |= (PARAM_FLAG_BUILD | PARAM_FLAG_FINAL);
                    }else{
                        m_TracerMap[m_CurTracerName].pipelineName = "TRACE";
                        std::cout << "TRACE" << std::endl;
                        params.flags |= PARAM_FLAG_BUILD;
                    }
                }
                else {
                    std::cout << "BUILD" << std::endl;
                    m_TracerMap[m_CurTracerName].pipelineName      = "BUILD";
                    params.flags |= PARAM_FLAG_NONE;
                }
            }
        }
        if (m_EnableGrid) {
            params.flags |= PARAM_FLAG_USE_GRID;
        }
    }
    stream->CopyMemoryToBuffer(m_TracerMap[m_CurTracerName].GetPipelineData().paramsBuffer.get(), { { &params, 0, sizeof(params) } });
    m_TracerMap[m_CurTracerName].Launch(stream, m_SceneData.config.width, m_SceneData.config.height);
    if (m_CurTracerName == "PGDEF") {
        stream->Synchronize();
        TraceEndPgDefPipeline();
    }
    if (m_CurTracerName != "DBG") {
        m_SamplesForAccum += m_SceneData.config.samples;
    }
}

 bool RTLibExtOPX7TestApplication::FinishTrace()
{
    if (m_EnableVis)
    {
        return m_GlfwWindow->ShouldClose() || (m_SamplesForAccum >= m_SceneData.config.maxSamples);
    }
    else
    {
        return (m_SamplesForAccum >= m_SceneData.config.maxSamples);
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
            m_TracerMap[m_CurTracerName].GetPipelineData().SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
            m_TracerMap[m_CurTracerName].GetPipelineData().shaderTable->UploadRaygenRecord();
        }
        if (m_SamplesForAccum % 100 == 99)
        {
            auto rnd = std::random_device();
            auto mt19937 = std::mt19937(rnd());
            auto seedData = std::vector<unsigned int>(m_SceneData.config.width * m_SceneData.config.height * sizeof(unsigned int));
            std::generate(std::begin(seedData), std::end(seedData), mt19937);
            m_Stream->CopyMemoryToBuffer(m_SeedBufferCUDA.get(), { { seedData.data(), 0, sizeof(seedData[0]) * std::size(seedData) } });
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
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_P )->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_P )->isPressed) {
            std::cout << "Save" << std::endl;
            auto        pixels     = std::vector<unsigned char>(m_SceneData.config.width * m_SceneData.config.height*4);
            auto     texResIdx = RTLib::Ext::GL::GLNatives::GetResId(m_FrameTextureGL.get());
            glGetTextureImage(texResIdx, 0, GL_RGBA, GL_UNSIGNED_BYTE,pixels.size(), pixels.data());
            auto image_path = m_SceneData.config.imagePath + "\\result_" + m_TimeStampString + ".png";
            stbi_write_png(image_path.c_str(), m_SceneData.config.width, m_SceneData.config.height, 4, pixels.data(), m_SceneData.config.width * 4);
            
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
    if (m_EnableVis)
    {
        auto ogl4Context = m_GlfwWindow->GetOpenGLContext();
        m_TimesForFrame = 0;
        auto beg = std::chrono::system_clock::now();
        auto frameBufferCUDA = m_FrameBufferCUGL->Map(stream);
        this->TracePipeline(stream, frameBufferCUDA);
        m_FrameBufferCUGL->Unmap(stream);
        stream->Synchronize();
        auto end = std::chrono::system_clock::now();
        m_TimesForFrame = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
        if (m_EventState.isResized)
        {
            glViewport(0, 0, m_SceneData.config.width, m_SceneData.config.height);
        }
        rtlib::test::RenderFrameGL(ogl4Context, m_RectRendererGL.get(), m_FrameBufferGL.get(), m_FrameTextureGL.get());
    }
    else
    {
        m_TimesForFrame = 0;
        auto beg = std::chrono::system_clock::now();
        this->TracePipeline(stream, m_FrameBufferCUDA.get());
        stream->Synchronize();
        auto end = std::chrono::system_clock::now();
        m_TimesForFrame = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
    }
    if (m_CurTracerName != "DBG") {
        m_TimesForAccum += m_TimesForFrame;
        if ((m_SamplesForAccum > 0) && (m_SamplesForAccum % m_SceneData.config.samplesPerSave == 0))
        {
            auto baseSavePath = std::filesystem::path(m_SceneData.config.imagePath).make_preferred() / m_TimeStampString;
            if (!std::filesystem::exists(baseSavePath))
            {
                std::filesystem::create_directory(baseSavePath);
                std::filesystem::copy_file(m_ScenePath, baseSavePath / "scene.json");
            }
            auto configData      = rtlib::test::ImageConfigData();
            configData.width     = m_SceneData.config.width;
            configData.height    = m_SceneData.config.height;
            configData.samples   = m_SamplesForAccum;
            configData.time      = m_TimesForAccum;
            configData.enableVis = m_EnableVis;
            configData.pngFilePath = baseSavePath.string() + "/result_" + m_CurTracerName + "_" + std::to_string(m_SamplesForAccum) + ".png";
            configData.binFilePath = baseSavePath.string() + "/result_" + m_CurTracerName + "_" + std::to_string(m_SamplesForAccum) + ".bin";
            configData.exrFilePath = baseSavePath.string() + "/result_" + m_CurTracerName + "_" + std::to_string(m_SamplesForAccum) + ".exr";
            {
                std::ofstream configFile (baseSavePath.string() + "/config_" + m_CurTracerName + "_" + std::to_string(m_SamplesForAccum) + ".json");
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
                std::ofstream imageBinFile(configData.binFilePath, std::ios::binary|std::ios::ate);
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
    }
}

void RTLibExtOPX7TestApplication::TraceBegPgDefPipeline()
 {
     if (m_Variables.GetBool("Started"))
     {
         auto sampleForBudget = m_Variables.GetUInt32("SampleForBudget");
         auto samplePerLaunch = m_Variables.GetUInt32("SamplePerLaunch");
         m_Variables.SetUInt32("SamplePerAll", 0);
         m_Variables.SetUInt32("SamplePerTmp", 0);
         m_Variables.SetUInt32("CurIteration", 0);
         m_Variables.SetBool("Launched", true);
         m_Variables.SetBool("Started", false);
         m_Variables.SetUInt32("SampleForRemain", ((sampleForBudget - 1 + samplePerLaunch) / samplePerLaunch) * samplePerLaunch);
         m_Variables.SetUInt32("SampleForPass", 0);
         this->m_SdTree->Clear();
         this->m_SdTree->Upload();
     }
     auto curIteration = m_Variables.GetUInt32("CurIteration");
     auto samplePerAll = m_Variables.GetUInt32("SamplePerAll");
     auto samplePerTmp = m_Variables.GetUInt32("SamplePerTmp");
     auto samplePerLaunch = m_Variables.GetUInt32("SamplePerLaunch");
     auto sampleForBudget = m_Variables.GetUInt32("SampleForBudget");
     auto ratioForBudget = m_Variables.GetFloat1("RatioForBudget");
     auto sampleForRemain = m_Variables.GetUInt32("SampleForRemain");
     auto sampleForPass = m_Variables.GetUInt32("SampleForPass");
     if (samplePerTmp == 0)
     {
         //CurIteration > 0 -> Reset
         sampleForRemain = sampleForRemain - sampleForPass;
         sampleForPass = std::min<uint32_t>(sampleForRemain, (1 << curIteration) * samplePerLaunch);
         if ((sampleForRemain - sampleForPass < 2 * sampleForPass) || (samplePerAll >= ratioForBudget * static_cast<float>(sampleForBudget)))
         {
             std::cout << "Final: this->m_Impl->m_SamplePerAll=" << samplePerAll << std::endl;
             sampleForPass = sampleForRemain;
         }
         /*Remain>Pass -> Not Final Iteration*/
         if (sampleForRemain > sampleForPass)
         {
             this->m_SdTree->Download();
             this->m_SdTree->Reset(curIteration, samplePerLaunch);
             this->m_SdTree->Upload();
         }
     }
     std::cout << "CurIteration: " << curIteration << " SamplePerTmp: " << samplePerTmp << std::endl;
     m_Variables.SetUInt32("SampleForRemain", sampleForRemain);
     m_Variables.SetUInt32("SampleForPass", sampleForPass);
 }

void RTLibExtOPX7TestApplication::TraceEndPgDefPipeline() {
     auto samplePerAll = m_Variables.GetUInt32("SamplePerAll");
     auto samplePerTmp = m_Variables.GetUInt32("SamplePerTmp");
     samplePerAll += m_SceneData.config.samples;
     samplePerTmp += m_SceneData.config.samples;
     m_Variables.SetUInt32("SamplePerAll", samplePerAll);
     m_Variables.SetUInt32("SamplePerTmp", samplePerTmp);

     auto sampleForBudget = m_Variables.GetUInt32("SampleForBudget");
     auto sampleForPass = m_Variables.GetUInt32("SampleForPass");
     auto curIteration = m_Variables.GetUInt32("CurIteration");
     if (samplePerTmp >= sampleForPass)
     {
         this->m_SdTree->Download();
         this->m_SdTree->Build();
         this->m_SdTree->Upload();

         curIteration++;
         m_Variables.SetUInt32("SamplePerTmp", 0);
         m_Variables.SetUInt32("CurIteration", curIteration);
     }
     if (samplePerAll >= sampleForBudget)
     {
         m_Variables.SetBool("Launched", false);
         m_Variables.SetBool("Started", false);
         m_Variables.SetUInt32("SamplePerAll", 0);
         m_Variables.SetBool("Finished", true);
     }
     else {
         m_Variables.SetBool("Finished", false);
     }
 }
