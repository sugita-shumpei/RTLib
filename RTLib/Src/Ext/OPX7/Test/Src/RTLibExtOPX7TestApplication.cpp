#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYEXR_IMPLEMENTATION
#include <RTLibExtOPX7TestApplication.h>

void RTLibExtOPX7TestApplication::cursorPosCallback(RTLib::Core::Window* window, double x, double y)
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
    m_TextureMap = m_SceneData.LoadTextureMap(m_Opx7Context.get());
    m_GeometryASMap = m_SceneData.BuildGeometryASs(m_Opx7Context.get(), accelBuildOptions);
    m_InstanceASMap = m_SceneData.BuildInstanceASs(m_Opx7Context.get(), accelBuildOptions, m_ShaderTableLayout.get(), m_GeometryASMap);
    auto aabb = rtlib::test::AABB();
    for (auto& instancePath : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instancePath);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
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
                reinterpret_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(uniqueRes.get())->Destroy();
                uniqueRes->extData.reset();
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
    m_HashBufferCUDA.Alloc(make_uint3(1024, 1024, 1024), 1024 * 1024 * 16);
    m_HashBufferCUDA.Upload(m_Opx7Context.get());
}

 void RTLibExtOPX7TestApplication::FreeGrids()
{
    m_HashBufferCUDA.dataGpuHandle->Destroy();
    m_HashBufferCUDA.checkSumGpuHandle->Destroy();
}

 void RTLibExtOPX7TestApplication::InitPtxString()
{
    m_PtxStringMap["SimpleKernel.ptx"] = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleKernel.ptx");
}

void RTLibExtOPX7TestApplication::InitDefPipeline()
{
    m_PipelineMap["DEF"].compileOptions.usesMotionBlur = false;
    m_PipelineMap["DEF"].compileOptions.traversableGraphFlags = 0;
    m_PipelineMap["DEF"].compileOptions.numAttributeValues = 3;
    m_PipelineMap["DEF"].compileOptions.numPayloadValues = 8;
    m_PipelineMap["DEF"].compileOptions.launchParamsVariableNames = "params";
    m_PipelineMap["DEF"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle;
    m_PipelineMap["DEF"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_PipelineMap["DEF"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_PipelineMap["DEF"].linkOptions.maxTraceDepth = 1;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_NONE;
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eMinimal;
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        moduleOptions.boundValueEntries.push_back({});
        moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
        moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
        moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
        m_PipelineMap["DEF"].LoadModule(m_Opx7Context.get(), "SimpleKernel.DEF", moduleOptions, m_PtxStringMap.at("SimpleKernel.ptx"));
    }
    m_PipelineMap["DEF"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.DEF", "__raygen__default");
    m_PipelineMap["DEF"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.DEF", "__exception__ep");
    m_PipelineMap["DEF"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.DEF", "__miss__radiance");
    m_PipelineMap["DEF"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.DEF", "__miss__occluded");
    m_PipelineMap["DEF"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.DEF", "__closesthit__radiance", "", "", "", "");
    m_PipelineMap["DEF"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.DEF", "__closesthit__occluded", "", "", "", "");
    m_PipelineMap["DEF"].InitPipeline(m_Opx7Context.get());
    m_PipelineMap["DEF"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
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
        m_PipelineMap["DEF"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" };
    auto raygenRecord = m_PipelineMap["DEF"].programGroupRG->GetRecord<RayGenData>();
    m_PipelineMap["DEF"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_PipelineMap["DEF"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
    m_PipelineMap["DEF"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    m_PipelineMap["DEF"].SetHostExceptionRecordTypeData(unsigned int());
    for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {
            for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshName);
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
                        m_PipelineMap["DEF"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride], hitgroupData);
                    }
                }
            }
        }
    }
    m_PipelineMap["DEF"].shaderTable->Upload();

    auto cssRG = m_PipelineMap["DEF"].programGroupRG->GetStackSize().cssRG;
    auto cssMS = std::max(m_PipelineMap["DEF"].programGroupMSs["SimpleKernel.Radiance"]->GetStackSize().cssMS, m_PipelineMap["DEF"].programGroupMSs["SimpleKernel.Occluded"]->GetStackSize().cssMS);
    auto cssCH = std::max(m_PipelineMap["DEF"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssCH, m_PipelineMap["DEF"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssCH);
    auto cssIS = std::max(m_PipelineMap["DEF"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssIS, m_PipelineMap["DEF"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssIS);
    auto cssAH = std::max(m_PipelineMap["DEF"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssAH, m_PipelineMap["DEF"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssAH);
    auto cssCCTree = static_cast<unsigned int>(0);
    auto cssCHOrMsPlusCCTree = std::max(cssMS, cssCH) + cssCCTree;
    auto continuationStackSizes = cssRG + cssCCTree +
        (std::max<unsigned int>(1, m_PipelineMap["DEF"].linkOptions.maxTraceDepth) - 1) * cssCHOrMsPlusCCTree +
        (std::min<unsigned int>(1, m_PipelineMap["DEF"].linkOptions.maxTraceDepth)) * std::max(cssCHOrMsPlusCCTree, cssIS + cssAH);

    m_PipelineMap["DEF"].handle->SetStackSize(0, 0, continuationStackSizes, m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {
        params.flags = PARAM_FLAG_NONE;
    }
    m_PipelineMap["DEF"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
}

void RTLibExtOPX7TestApplication::InitNeePipeline()
{
    m_PipelineMap["NEE"].compileOptions.usesMotionBlur = false;
    m_PipelineMap["NEE"].compileOptions.traversableGraphFlags = 0;
    m_PipelineMap["NEE"].compileOptions.numAttributeValues = 3;
    m_PipelineMap["NEE"].compileOptions.numPayloadValues = 8;
    m_PipelineMap["NEE"].compileOptions.launchParamsVariableNames = "params";
    m_PipelineMap["NEE"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle;
    m_PipelineMap["NEE"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_PipelineMap["NEE"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_PipelineMap["NEE"].linkOptions.maxTraceDepth = 2;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_NEE;
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eMinimal;
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        moduleOptions.boundValueEntries.push_back({});
        moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
        moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
        moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
        m_PipelineMap["NEE"].LoadModule(m_Opx7Context.get(), "SimpleKernel.NEE", moduleOptions, m_PtxStringMap.at("SimpleKernel.ptx"));
    }
    m_PipelineMap["NEE"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.NEE", "__raygen__default");
    m_PipelineMap["NEE"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.NEE", "__exception__ep");
    m_PipelineMap["NEE"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.NEE", "__miss__radiance");
    m_PipelineMap["NEE"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.NEE", "__miss__occluded");
    m_PipelineMap["NEE"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.NEE", "__closesthit__radiance", "", "", "", "");
    m_PipelineMap["NEE"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.NEE", "__closesthit__occluded", "", "", "", "");
    m_PipelineMap["NEE"].InitPipeline(m_Opx7Context.get());
    m_PipelineMap["NEE"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
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
        m_PipelineMap["NEE"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" };
    auto raygenRecord = m_PipelineMap["NEE"].programGroupRG->GetRecord<RayGenData>();
    m_PipelineMap["NEE"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_PipelineMap["NEE"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
    m_PipelineMap["NEE"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    m_PipelineMap["NEE"].SetHostExceptionRecordTypeData(unsigned int());
    for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {

            for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshName);
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
                            hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                            hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
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
                        m_PipelineMap["NEE"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride], hitgroupData);
                    }
                }
            }
        }
    }
    m_PipelineMap["NEE"].shaderTable->Upload();

    auto cssRG = m_PipelineMap["NEE"].programGroupRG->GetStackSize().cssRG;
    auto cssMS = std::max(m_PipelineMap["NEE"].programGroupMSs["SimpleKernel.Radiance"]->GetStackSize().cssMS, m_PipelineMap["NEE"].programGroupMSs["SimpleKernel.Occluded"]->GetStackSize().cssMS);
    auto cssCH = std::max(m_PipelineMap["NEE"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssCH, m_PipelineMap["NEE"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssCH);
    auto cssIS = std::max(m_PipelineMap["NEE"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssIS, m_PipelineMap["NEE"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssIS);
    auto cssAH = std::max(m_PipelineMap["NEE"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssAH, m_PipelineMap["NEE"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssAH);
    auto cssCCTree = static_cast<unsigned int>(0);
    auto cssCHOrMsPlusCCTree = std::max(cssMS, cssCH) + cssCCTree;
    auto continuationStackSizes = cssRG + cssCCTree +
        (std::max<unsigned int>(1, m_PipelineMap["NEE"].linkOptions.maxTraceDepth) - 1) * cssCHOrMsPlusCCTree +
        (std::min<unsigned int>(1, m_PipelineMap["NEE"].linkOptions.maxTraceDepth)) * std::max(cssCHOrMsPlusCCTree, cssIS + cssAH);

    m_PipelineMap["NEE"].handle->SetStackSize(0, 0, continuationStackSizes, m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {

        params.flags = PARAM_FLAG_NEE;
    }
    m_PipelineMap["NEE"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
}

void RTLibExtOPX7TestApplication::InitRisPipeline()
{
    m_PipelineMap["RIS"].compileOptions.usesMotionBlur = false;
    m_PipelineMap["RIS"].compileOptions.traversableGraphFlags = 0;
    m_PipelineMap["RIS"].compileOptions.numAttributeValues = 3;
    m_PipelineMap["RIS"].compileOptions.numPayloadValues = 8;
    m_PipelineMap["RIS"].compileOptions.launchParamsVariableNames = "params";
    m_PipelineMap["RIS"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle;
    m_PipelineMap["RIS"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_PipelineMap["RIS"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_PipelineMap["RIS"].linkOptions.maxTraceDepth = 2;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_NEE|PARAM_FLAG_RIS;
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eMinimal;
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        moduleOptions.boundValueEntries.push_back({});
        moduleOptions.boundValueEntries.front().pipelineParamOffsetInBytes = offsetof(Params, flags);
        moduleOptions.boundValueEntries.front().boundValuePtr = &flags;
        moduleOptions.boundValueEntries.front().sizeInBytes = sizeof(flags);
        m_PipelineMap["RIS"].LoadModule(m_Opx7Context.get(), "SimpleKernel.RIS", moduleOptions, m_PtxStringMap.at("SimpleKernel.ptx"));
    }
    m_PipelineMap["RIS"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.RIS", "__raygen__default");
    m_PipelineMap["RIS"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.RIS", "__exception__ep");
    m_PipelineMap["RIS"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.RIS", "__miss__radiance");
    m_PipelineMap["RIS"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.RIS", "__miss__occluded");
    m_PipelineMap["RIS"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Radiance", "SimpleKernel.RIS", "__closesthit__radiance", "", "", "", "");
    m_PipelineMap["RIS"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Occluded", "SimpleKernel.RIS", "__closesthit__occluded", "", "", "", "");
    m_PipelineMap["RIS"].InitPipeline(m_Opx7Context.get());
    m_PipelineMap["RIS"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
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
        m_PipelineMap["RIS"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Radiance",
        "SimpleKernel.Occluded" };
    auto raygenRecord = m_PipelineMap["RIS"].programGroupRG->GetRecord<RayGenData>();
    m_PipelineMap["RIS"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_PipelineMap["RIS"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
    m_PipelineMap["RIS"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
    m_PipelineMap["RIS"].SetHostExceptionRecordTypeData(unsigned int());
    for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {

            for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshName);
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
                            hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                            hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
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
                        m_PipelineMap["RIS"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride], hitgroupData);
                    }
                }
            }
        }
    }
    m_PipelineMap["RIS"].shaderTable->Upload();

    auto cssRG = m_PipelineMap["RIS"].programGroupRG->GetStackSize().cssRG;
    auto cssMS = std::max(m_PipelineMap["RIS"].programGroupMSs["SimpleKernel.Radiance"]->GetStackSize().cssMS, m_PipelineMap["RIS"].programGroupMSs["SimpleKernel.Occluded"]->GetStackSize().cssMS);
    auto cssCH = std::max(m_PipelineMap["RIS"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssCH, m_PipelineMap["RIS"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssCH);
    auto cssIS = std::max(m_PipelineMap["RIS"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssIS, m_PipelineMap["RIS"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssIS);
    auto cssAH = std::max(m_PipelineMap["RIS"].programGroupHGs["SimpleKernel.Radiance"]->GetStackSize().cssAH, m_PipelineMap["RIS"].programGroupHGs["SimpleKernel.Occluded"]->GetStackSize().cssAH);
    auto cssCCTree = static_cast<unsigned int>(0);
    auto cssCHOrMsPlusCCTree = std::max(cssMS, cssCH) + cssCCTree;
    auto continuationStackSizes = cssRG + cssCCTree +
        (std::max<unsigned int>(1, m_PipelineMap["RIS"].linkOptions.maxTraceDepth) - 1) * cssCHOrMsPlusCCTree +
        (std::min<unsigned int>(1, m_PipelineMap["RIS"].linkOptions.maxTraceDepth)) * std::max(cssCHOrMsPlusCCTree, cssIS + cssAH);

    m_PipelineMap["RIS"].handle->SetStackSize(0, 0, continuationStackSizes, m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {

        params.flags = PARAM_FLAG_RIS|PARAM_FLAG_NEE;
    }
    m_PipelineMap["RIS"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
}

void RTLibExtOPX7TestApplication::InitDbgPipeline() {
    m_PipelineMap["DBG"].compileOptions.usesMotionBlur = false;
    m_PipelineMap["DBG"].compileOptions.traversableGraphFlags = 0;
    m_PipelineMap["DBG"].compileOptions.numAttributeValues = 3;
    m_PipelineMap["DBG"].compileOptions.numPayloadValues = 8;
    m_PipelineMap["DBG"].compileOptions.launchParamsVariableNames = "params";
    m_PipelineMap["DBG"].compileOptions.usesPrimitiveTypeFlags = RTLib::Ext::OPX7::OPX7PrimitiveTypeFlagsTriangle;
    m_PipelineMap["DBG"].compileOptions.exceptionFlags = RTLib::Ext::OPX7::OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
    m_PipelineMap["DBG"].linkOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
    m_PipelineMap["DBG"].linkOptions.maxTraceDepth = 1;
    {
        auto moduleOptions = RTLib::Ext::OPX7::OPX7ModuleCompileOptions{};
        unsigned int flags = PARAM_FLAG_NEE;
        moduleOptions.optLevel = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
        moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eMinimal;
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.payloadTypes = {};
        m_PipelineMap["DBG"].LoadModule(m_Opx7Context.get(), "SimpleKernel.DBG", moduleOptions, m_PtxStringMap.at("SimpleKernel.ptx"));
    }
    m_PipelineMap["DBG"].SetProgramGroupRG(m_Opx7Context.get(), "SimpleKernel.DBG", "__raygen__debug");
    m_PipelineMap["DBG"].SetProgramGroupEP(m_Opx7Context.get(), "SimpleKernel.DBG", "__exception__ep");
    m_PipelineMap["DBG"].SetProgramGroupMS(m_Opx7Context.get(), "SimpleKernel.Debug", "SimpleKernel.DBG", "__miss__debug");
    m_PipelineMap["DBG"].SetProgramGroupHG(m_Opx7Context.get(), "SimpleKernel.Debug", "SimpleKernel.DBG", "__closesthit__debug", "", "", "", "");
    m_PipelineMap["DBG"].InitPipeline(m_Opx7Context.get());
    m_PipelineMap["DBG"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>();
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
        m_PipelineMap["DBG"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto programGroupHGNames = std::vector<std::string>{
        "SimpleKernel.Debug",
        "SimpleKernel.Debug" };
    auto raygenRecord = m_PipelineMap["DBG"].programGroupRG->GetRecord<RayGenData>();
    m_PipelineMap["DBG"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
    m_PipelineMap["DBG"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Debug", MissData{});
    m_PipelineMap["DBG"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Debug", MissData{});
    m_PipelineMap["DBG"].SetHostExceptionRecordTypeData(unsigned int());
    for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
    {
        auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
        auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
        auto geometryAS = instanceData->GetDwGeometryAS();
        if (geometryAS)
        {

            for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
            {
                auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                for (auto& [meshName, meshData] : geometry.meshes)
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(meshName);
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
                            hitgroupData.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                            hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
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
                        m_PipelineMap["DBG"].SetHostHitgroupRecordTypeData(desc.recordOffset + i, programGroupHGNames[i % desc.recordStride], hitgroupData);
                    }
                }
            }
        }
    }
    m_PipelineMap["DBG"].shaderTable->Upload();

    auto cssRG = m_PipelineMap["DBG"].programGroupRG->GetStackSize().cssRG;
    auto cssMS = std::max(m_PipelineMap["DBG"].programGroupMSs["SimpleKernel.Debug"]->GetStackSize().cssMS, m_PipelineMap["DBG"].programGroupMSs["SimpleKernel.Debug"]->GetStackSize().cssMS);
    auto cssCH = std::max(m_PipelineMap["DBG"].programGroupHGs["SimpleKernel.Debug"]->GetStackSize().cssCH, m_PipelineMap["DBG"].programGroupHGs["SimpleKernel.Debug"]->GetStackSize().cssCH);
    auto cssIS = std::max(m_PipelineMap["DBG"].programGroupHGs["SimpleKernel.Debug"]->GetStackSize().cssIS, m_PipelineMap["DBG"].programGroupHGs["SimpleKernel.Debug"]->GetStackSize().cssIS);
    auto cssAH = std::max(m_PipelineMap["DBG"].programGroupHGs["SimpleKernel.Debug"]->GetStackSize().cssAH, m_PipelineMap["DBG"].programGroupHGs["SimpleKernel.Debug"]->GetStackSize().cssAH);
    auto cssCCTree = static_cast<unsigned int>(0);
    auto cssCHOrMsPlusCCTree = std::max(cssMS, cssCH) + cssCCTree;
    auto continuationStackSizes = cssRG + cssCCTree +
        (std::max<unsigned int>(1, m_PipelineMap["DBG"].linkOptions.maxTraceDepth) - 1) * cssCHOrMsPlusCCTree +
        (std::min<unsigned int>(1, m_PipelineMap["DBG"].linkOptions.maxTraceDepth)) * std::max(cssCHOrMsPlusCCTree, cssIS + cssAH);

    m_PipelineMap["DBG"].handle->SetStackSize(0, 0, continuationStackSizes, m_ShaderTableLayout->GetMaxTraversableDepth());
    auto params = Params();
    {

        params.flags = PARAM_FLAG_NEE;
    }
    m_PipelineMap["DBG"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
}


 void RTLibExtOPX7TestApplication::FreePipelines()
{
    for (auto& [name, pipeline] : m_PipelineMap)
    {
        pipeline.Free();
    }
    m_PipelineMap.clear();
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
    m_GlfwWindow->SetCursorPosCallback(cursorPosCallback);
}

 void RTLibExtOPX7TestApplication::TracePipeline(RTLib::Ext::CUDA::CUDAStream* stream, RTLib::Ext::CUDA::CUDABuffer* frameBuffer)
{
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

        if (m_CurPipelineName == "DEF")
        {
            params.flags = PARAM_FLAG_NONE;
        }
        if (m_CurPipelineName == "NEE")
        {
            params.flags = PARAM_FLAG_NEE;
        }
        if (m_CurPipelineName == "RIS") {
            params.flags = PARAM_FLAG_NEE| PARAM_FLAG_RIS ;
        }
    }
    stream->CopyMemoryToBuffer(m_PipelineMap[m_CurPipelineName].paramsBuffer.get(), { { &params, 0, sizeof(params) } });
    m_PipelineMap[m_CurPipelineName].Launch(stream, m_SceneData.config.width, m_SceneData.config.height);
    if (m_CurPipelineName != "DBG") {

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
            this->UpdateTimeStamp();
        }
        if (m_EventState.isMovedCamera)
        {
            m_PipelineMap[m_CurPipelineName].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
            m_PipelineMap[m_CurPipelineName].shaderTable->UploadRaygenRecord();
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
    if (m_EnableVis)
    {
        m_EventState = rtlib::test::EventState();

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
            m_PrvPipelineName = m_CurPipelineName;
            if (m_PrvPipelineName != "DBG")
            {
                std::cout << "State Change" << std::endl;
                m_CurPipelineName = "DBG";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F2)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F2)->isPressed)
        {
            m_PrvPipelineName = m_CurPipelineName;
            if (m_PrvPipelineName != "DEF")
            {
                std::cout << "State Change" << std::endl;
                m_CurPipelineName = "DEF";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F3)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F3)->isPressed)
        {
            m_PrvPipelineName = m_CurPipelineName;
            if (m_PrvPipelineName != "NEE")
            {
                std::cout << "State Change" << std::endl;
                m_CurPipelineName = "NEE";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
            }
        }
        if (m_KeyBoardManager->GetState(GLFW_KEY_F4)->isUpdated &&
            m_KeyBoardManager->GetState(GLFW_KEY_F4)->isPressed)
        {
            m_PrvPipelineName = m_CurPipelineName;
            if (m_PrvPipelineName != "RIS")
            {
                std::cout << "State Change" << std::endl;
                m_CurPipelineName = "RIS";
                m_EventState.isMovedCamera = true;
                m_EventState.isClearFrame = true;
            }
        }


        if (m_CurPipelineName == "DBG") {
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
        auto frameBufferCUDA = m_FrameBufferCUGL->Map(stream);
        this->TracePipeline(stream, frameBufferCUDA);
        m_FrameBufferCUGL->Unmap(stream);
        stream->Synchronize();
        if (m_EventState.isResized)
        {
            glViewport(0, 0, m_SceneData.config.width, m_SceneData.config.height);
        }
        rtlib::test::RenderFrameGL(ogl4Context, m_RectRendererGL.get(), m_FrameBufferGL.get(), m_FrameTextureGL.get());
    }
    else
    {
        this->TracePipeline(stream, m_FrameBufferCUDA.get());
        stream->Synchronize();
    }
    if (m_CurPipelineName != "DBG") {

        if ((m_SamplesForAccum > 0) && (m_SamplesForAccum % m_SceneData.config.samplesPerSave == 0))
        {
            auto baseSavePath = std::filesystem::path(m_SceneData.config.imagePath).make_preferred() / m_TimeStampString;
            if (!std::filesystem::exists(baseSavePath))
            {
                std::filesystem::create_directory(baseSavePath);
                std::filesystem::copy_file(m_ScenePath, baseSavePath / "scene.json");
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
                std::string saveExrPath = baseSavePath.string() + "/result_" + m_CurPipelineName + "_" + std::to_string(m_SamplesForAccum) + ".exr";
                rtlib::test::SaveExrImage(saveExrPath.c_str(), m_SceneData.config.width, m_SceneData.config.height, hdr_image_data);
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
                std::string savePngPath = baseSavePath.string() + "/result_" + m_CurPipelineName + "_" + std::to_string(m_SamplesForAccum) + ".png";
                rtlib::test::SavePngImage(savePngPath.c_str(), m_SceneData.config.width, m_SceneData.config.height, png_image_data);
            }
        }
    }
}
