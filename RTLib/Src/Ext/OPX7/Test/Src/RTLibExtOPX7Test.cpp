#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLibExtOPX7Test.h>

class RTLibExtOPX7TestApplication
{

};
struct WindowState
{
    float curTime    = 0.0f;
    float delTime    = 0.0f;
    float2 curCurPos = {};
    float2 delCurPos = {};
};
static void cursorPosCallback(RTLib::Core::Window *window, double x, double y)
{
    auto pWindowState = reinterpret_cast<WindowState *>(window->GetUserPointer());
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
int OnlineSample()
{
    using namespace RTLib::Ext::CUDA;
    using namespace RTLib::Ext::OPX7;
    using namespace RTLib::Ext::GL;
    using namespace RTLib::Ext::CUGL;
    using namespace RTLib::Ext::GLFW;

    auto sceneJson = nlohmann::json();
    {
        std::ifstream sceneJsonFile(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", std::ios::binary);
        if (sceneJsonFile.is_open())
        {
            sceneJson = nlohmann::json::parse(sceneJsonFile);
            sceneJsonFile.close();
        }
        else
        {
            sceneJson = rtlib::test::GetDefaultSceneJson();
        }
    }
    auto sceneData = sceneJson.get<rtlib::test::SceneData>();
    int  width             = sceneData.config.width;
    int  height            = sceneData.config.height;
    int  samplesForLaunch  = sceneData.config.samples;
    int  maxSamples        = sceneData.config.maxSamples;
    int  samplesForAccum   = 0;
    auto& cameraController = sceneData.cameraController;
    auto& objAssetLoader   = sceneData.objAssetManager;
    auto&     worldData        = sceneData.world;
    try
    {
        auto glfwContext = std::unique_ptr<GLFWContext>(GLFWContext::New());
        glfwContext->Initialize();

        auto window = std::unique_ptr<GL::GLFWOpenGLWindow>(rtlib::test::CreateGLFWWindow(glfwContext.get(), width, height, "title"));
        // contextはcopy/move不可
        auto opx7Context = std::unique_ptr<OPX7Context>();
        {
            auto contextCreateDesc = OPX7ContextCreateDesc();
            {
                contextCreateDesc.validationMode = OPX7ContextValidationMode::eALL;
                contextCreateDesc.level = OPX7ContextValidationLogLevel::ePrint;
            }
            opx7Context = std::make_unique<OPX7Context>(contextCreateDesc);
        }
        opx7Context->Initialize();

        auto ogl4Context = window->GetOpenGLContext();

        auto shaderTableLayout = rtlib::test::GetShaderTableLayout(worldData, RAY_TYPE_COUNT);
        auto accelBuildOptions = OptixAccelBuildOptions();
        {
            accelBuildOptions.buildFlags    = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accelBuildOptions.motionOptions = {};
            accelBuildOptions.operation     = OPTIX_BUILD_OPERATION_BUILD;
        }
        sceneData.InitExtData(opx7Context.get());

        auto textures = std::unordered_map<std::string, rtlib::test::TextureData>();
        {
            std::unordered_set< std::string> texturePathSet = {};
            for (auto& [geometryName, geometryData] : sceneData.world.geometryObjModels)
            {
                for (auto& [meshName, meshData]: geometryData.meshes)
                {
                    for (auto& material : meshData.materials)
                    {
                        if (material.HasString("diffTex"))
                        {
                            if (material.GetString("diffTex")!="") {
                                texturePathSet.insert(material.GetString("diffTex"));
                            }
                        }
                        if (material.HasString("specTex"))
                        {
                            if (material.GetString("specTex") != "") {
                                texturePathSet.insert(material.GetString("specTex"));
                            }
                        }
                        if (material.HasString("emitTex"))
                        {
                            if (material.GetString("emitTex") != "") {
                                texturePathSet.insert(material.GetString("emitTex"));
                            }
                        }
                    }
                }
            }
            for (auto& texturePath : texturePathSet)
            {
                textures[texturePath].LoadFromPath(opx7Context.get(), texturePath);
            }
            textures["White"] = rtlib::test::TextureData::White(opx7Context.get());
            textures["Black"] = rtlib::test::TextureData::Black(opx7Context.get());
        }
        auto geometryASs  = sceneData.BuildGeometryASs(opx7Context.get(), accelBuildOptions);
        auto instBuffers  = std::unordered_map<std::string, std::unique_ptr<CUDABuffer>>();
        auto topLevelAS   = rtlib::test::AccelerationStructureData();

        {
            auto instIndices = std::unordered_map<std::string, unsigned int>();
            {
                unsigned int i = 0;
                for (auto& [name, instance] : worldData.instances) {
                    instIndices[name] = i;
                    ++i;
                }
            }
            auto& rootInstanceAS = worldData.instanceASs["Root"];
            auto  opx7Instances  = std::vector<OptixInstance>();
            {
                opx7Instances.reserve(rootInstanceAS.instances.size());
                for (auto& instanceName : rootInstanceAS.instances)
                {
                    auto& instanceData = worldData.instances[instanceName];
                    auto opx7Instance   = OptixInstance();
                    opx7Instance.traversableHandle = geometryASs.at(instanceData.base).handle;
                    opx7Instance.visibilityMask    = 255;
                    opx7Instance.instanceId        = instIndices[instanceName];
                    opx7Instance.sbtOffset         = shaderTableLayout->GetDesc("Root/" + instanceName).recordOffset;
                    opx7Instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
                    auto transforms = instanceData.transform;
                    std::memcpy(opx7Instance.transform, transforms.data(), transforms.size() * sizeof(float));
                    opx7Instances.push_back(opx7Instance);
                }
            }

            instBuffers["Root"] = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer(
                CUDABufferCreateDesc{
                    CUDAMemoryFlags::eDefault,
                    sizeof(OptixInstance) * opx7Instances.size(),
                    opx7Instances.data()
                })
            );

            {
                auto buildInputs = std::vector<OptixBuildInput>(1);
                buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
                buildInputs[0].instanceArray.instances = CUDANatives::GetCUdeviceptr(instBuffers["Root"].get());
                buildInputs[0].instanceArray.numInstances = opx7Instances.size();

                topLevelAS = OPX7Natives::BuildAccelerationStructure(opx7Context.get(), accelBuildOptions, buildInputs);
            }
        }

        auto pipelineCompileOptions = OPX7PipelineCompileOptions{};
        {
            pipelineCompileOptions.usesMotionBlur            = false;
            pipelineCompileOptions.traversableGraphFlags     = OPX7TraversableGraphFlagsAllowSingleLevelInstancing;
            pipelineCompileOptions.numAttributeValues        = 3;
            pipelineCompileOptions.numPayloadValues          = 8;
            pipelineCompileOptions.launchParamsVariableNames = "params";
            pipelineCompileOptions.usesPrimitiveTypeFlags    = OPX7PrimitiveTypeFlagsTriangle;
            pipelineCompileOptions.exceptionFlags = OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
        }
        // contextはcopy不可
        auto moduleCreateDesc = OPX7ModuleCreateDesc{};
        {
            moduleCreateDesc.ptxBinary = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleKernel.ptx");
            moduleCreateDesc.pipelineOptions = pipelineCompileOptions;
            moduleCreateDesc.moduleOptions.optLevel = OPX7CompileOptimizationLevel::eDefault;
            moduleCreateDesc.moduleOptions.debugLevel = OPX7CompileDebugLevel::eMinimal;
            moduleCreateDesc.moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCreateDesc.moduleOptions.payloadTypes = {};
            moduleCreateDesc.moduleOptions.boundValueEntries = {};
        }
        auto module = std::unique_ptr<RTLib::Ext::OPX7::OPX7Module>(opx7Context->CreateOPXModule(moduleCreateDesc));
        auto raygPG = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Raygen(  {module.get(), "__raygen__rg"})));
        auto missPGForRadiance = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Miss({ module.get(), "__miss__radiance" })));
        auto missPGForOccluded = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Miss({ module.get(), "__miss__occluded" })));
        auto hitgPGForRadiance = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Hitgroup({module.get(), "__closesthit__radiance"})));
        auto hitgPGForOccluded = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Hitgroup({ module.get(),"__closesthit__occluded" })));
        auto pipelineCreateDesc = OPX7PipelineCreateDesc{};
        {
            pipelineCreateDesc.linkOptions.maxTraceDepth = 1;
            pipelineCreateDesc.linkOptions.debugLevel    = OPX7CompileDebugLevel::eMinimal;
            pipelineCreateDesc.compileOptions = pipelineCompileOptions;
            pipelineCreateDesc.programGroups  = {
                raygPG.get(), 
                missPGForRadiance.get(), 
                missPGForOccluded.get(),
                hitgPGForRadiance.get(),
                hitgPGForOccluded.get()
            };
        }
        auto pipeline = std::unique_ptr<OPX7Pipeline>(opx7Context->CreateOPXPipeline(pipelineCreateDesc));

        auto shaderTable = std::unique_ptr<OPX7ShaderTable>();
        {

            auto shaderTableDesc = OPX7ShaderTableCreateDesc();
            {

                shaderTableDesc.raygenRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
                shaderTableDesc.missRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
                shaderTableDesc.missRecordCount = shaderTableLayout->GetRecordStride();
                shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
                shaderTableDesc.hitgroupRecordCount = shaderTableLayout->GetRecordCount();
            }
            shaderTable = std::unique_ptr<OPX7ShaderTable>(opx7Context->CreateOPXShaderTable(shaderTableDesc));
        }
        {
            {
                auto raygenRecord = raygPG->GetRecord<RayGenData>();
                rtlib::test::SetRaygenData(raygenRecord, sceneData.GetCamera());
                shaderTable->SetHostRaygenRecordTypeData<RayGenData>(raygenRecord);
            }
            {
                auto missRecord = missPGForRadiance->GetRecord<MissData>();
                missRecord.data.bgColor = float4{ 0.005f,0.005f,0.005f,0.005f };
                shaderTable->SetHostMissRecordTypeData(RAY_TYPE_RADIANCE , missRecord);
                shaderTable->SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, missPGForOccluded->GetRecord<MissData>());
            }
            for (auto &instanceName : worldData.instanceASs["Root"].instances)
            {
                auto& instanceData = worldData.instances.at(instanceName);
                if (instanceData.asType == "Geometry")
                {
                    for (auto &geometryName : worldData.geometryASs[instanceData.base].geometries)
                    {
                        auto& geometry = worldData.geometryObjModels[geometryName];
                        auto objAsset  = objAssetLoader.GetAsset(geometry.base);
                        for (auto& [meshName,meshData] : geometry.meshes) {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshName);
                            auto desc = shaderTableLayout->GetDesc("Root/" + instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupRecord = RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>();
                                if (i % desc.recordStride == RAY_TYPE_RADIANCE) {
                                    hitgroupRecord = hitgPGForRadiance->GetRecord<HitgroupData>();
                                }
                                if (i % desc.recordStride == RAY_TYPE_OCCLUDED) {
                                    hitgroupRecord = hitgPGForOccluded->GetRecord<HitgroupData>();
                                }

                                {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupRecord.data.vertices = reinterpret_cast<float3*>(CUDANatives::GetCUdeviceptr(extSharedData->GetVertexBufferView()));
                                    hitgroupRecord.data.indices  = reinterpret_cast<uint3*>( CUDANatives::GetCUdeviceptr(extUniqueData->GetTriIdxBufferView()));
                                    hitgroupRecord.data.texCrds  = reinterpret_cast<float2*>(CUDANatives::GetCUdeviceptr(extSharedData->GetTexCrdBufferView()));
                                    hitgroupRecord.data.diffuse  = material.GetFloat3As<float3>("diffCol");
                                    hitgroupRecord.data.emission = material.GetFloat3As<float3>("emitCol");
                                    hitgroupRecord.data.specular = material.GetFloat3As<float3>("specCol");
                                    auto diffTexStr = material.GetString("diffTex");
                                    if (diffTexStr =="") {
                                        diffTexStr = "White";
                                    }
                                    auto specTexStr = material.GetString("specTex");
                                    if (specTexStr == "") {
                                        specTexStr = "White";
                                    }
                                    auto emitTexStr = material.GetString("emitTex");
                                    if (emitTexStr == "") {
                                        emitTexStr = "White";
                                    }
                                    hitgroupRecord.data.diffuseTex  = RTLib::Ext::CUDA::CUDANatives::GetCUtexObject(textures.at(diffTexStr).handle.get());
                                    hitgroupRecord.data.specularTex = RTLib::Ext::CUDA::CUDANatives::GetCUtexObject(textures.at(specTexStr).handle.get());
                                    hitgroupRecord.data.emissionTex = RTLib::Ext::CUDA::CUDANatives::GetCUtexObject(textures.at(emitTexStr).handle.get());
                                }
                                shaderTable->SetHostHitgroupRecordTypeData(desc.recordOffset + i, hitgroupRecord);
                            }
                        }
                        
                    }
                }
            }
        }
        shaderTable->Upload();

        auto  seedBufferCUDA = std::unique_ptr<CUDABuffer>();
        {
            auto mt19937 = std::mt19937();
            auto seedData = std::vector<unsigned int>(width * height * sizeof(unsigned int));
            std::generate(std::begin(seedData), std::end(seedData), mt19937);
            seedBufferCUDA = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer(
                {CUDAMemoryFlags::eDefault,seedData.size() * sizeof(seedData[0]),seedData.data()}
            ));
        }

        auto accumBufferCUDA = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer(  { CUDAMemoryFlags::eDefault,width* height * sizeof(float)*3,nullptr }  ));
        auto frameBufferGL   = std::unique_ptr<GLBuffer  >(ogl4Context->CreateBuffer(GLBufferCreateDesc{sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr}));
        auto frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
        auto frameTextureGL  = std::unique_ptr<GLTexture>();
        {
            auto frameTextureDesc = RTLib::Ext::GL::GLTextureCreateDesc();
            frameTextureDesc.image.imageType     = RTLib::Ext::GL::GLImageType::e2D;
            frameTextureDesc.image.extent.width  = width;
            frameTextureDesc.image.extent.height = height;
            frameTextureDesc.image.extent.depth  = 0;
            frameTextureDesc.image.arrayLayers   = 0;
            frameTextureDesc.image.mipLevels     = 1;
            frameTextureDesc.image.format        = RTLib::Ext::GL::GLFormat::eRGBA8;
            frameTextureDesc.sampler.magFilter   = RTLib::Core::FilterMode::eLinear;
            frameTextureDesc.sampler.minFilter   = RTLib::Core::FilterMode::eLinear;

            frameTextureGL = std::unique_ptr<GLTexture>(ogl4Context->CreateTexture(frameTextureDesc));
        }
        auto rectRendererGL = std::unique_ptr<GLRectRenderer>(ogl4Context->CreateRectRenderer({1, false, true}));

        auto paramsBuffer = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer({CUDAMemoryFlags::eDefault, static_cast<size_t>(sizeof(Params)), nullptr}));
        auto stream = std::unique_ptr<CUDAStream>(opx7Context->CreateStream());

        auto isUpdated = false;
        auto isResized = false;
        auto isMovedCamera = false;

        auto windowState = WindowState();
        {
            window->Show();
            window->SetResizable(true);
            window->SetUserPointer(&windowState);
            window->SetCursorPosCallback(cursorPosCallback);
        }
        while (!window->ShouldClose())
        {
            if (samplesForAccum >= maxSamples) {
                break;
            }
            {
                if (isResized    )
                {
                    frameBufferCUGL->Destroy();
                    frameBufferGL->Destroy();
                    frameTextureGL->Destroy();
                    frameBufferGL   = std::unique_ptr<GLBuffer>(ogl4Context->CreateBuffer(GLBufferCreateDesc{sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr}));
                    frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
                    {
                        auto frameTextureDesc = RTLib::Ext::GL::GLTextureCreateDesc();
                        frameTextureDesc.image.imageType = RTLib::Ext::GL::GLImageType::e2D;
                        frameTextureDesc.image.extent.width = width;
                        frameTextureDesc.image.extent.height = height;
                        frameTextureDesc.image.extent.depth = 0;
                        frameTextureDesc.image.arrayLayers = 0;
                        frameTextureDesc.image.mipLevels = 1;
                        frameTextureDesc.image.format = RTLib::Ext::GL::GLFormat::eRGBA8;
                        frameTextureDesc.sampler.magFilter = RTLib::Core::FilterMode::eLinear;
                        frameTextureDesc.sampler.minFilter = RTLib::Core::FilterMode::eLinear;

                        frameTextureGL = std::unique_ptr<GLTexture>(ogl4Context->CreateTexture(frameTextureDesc));
                    }
                }
                if (isUpdated    ) {
                    auto zeroClearData = std::vector<float>(width * height * 3, 0.0f);
                    opx7Context->CopyMemoryToBuffer(accumBufferCUDA.get(), {{zeroClearData.data(), 0, sizeof(zeroClearData[0])* zeroClearData.size()}});
                    samplesForAccum = 0;
                }
                if (isMovedCamera)
                {
                    auto raygenRecord = raygPG->GetRecord<RayGenData>();
                    rtlib::test::SetRaygenData(raygenRecord, sceneData.GetCamera());
                    shaderTable->SetHostRaygenRecordTypeData<RayGenData>(raygenRecord);
                    shaderTable->UploadRaygenRecord();
                }
            }
            glFinish();
            {

                auto frameBufferCUDA = frameBufferCUGL->Map(stream.get());
                /*RayTrace*/
                {
                    auto params = Params();
                    {
                        params.accumBuffer      = reinterpret_cast<float3*>(CUDANatives::GetCUdeviceptr(accumBufferCUDA.get()));
                        params.seedBuffer       = reinterpret_cast<unsigned int*>(CUDANatives::GetCUdeviceptr( seedBufferCUDA.get()));
                        params.frameBuffer      = reinterpret_cast<uchar4 *>(CUDANatives::GetCUdeviceptr(frameBufferCUDA));
                        params.width            = width;
                        params.height           = height;
                        params.samplesForAccum  = samplesForAccum;
                        params.samplesForLaunch = samplesForLaunch;
                        params.gasHandle        = topLevelAS.handle;
                    }
                    stream->CopyMemoryToBuffer(paramsBuffer.get(), {{&params, 0, sizeof(params)}});
                    pipeline->Launch(stream.get(), CUDABufferView(paramsBuffer.get(), 0, paramsBuffer->GetSizeInBytes()), shaderTable.get(), width, height, 1);
                    samplesForAccum += samplesForLaunch;
                }
                frameBufferCUGL->Unmap(stream.get());
                stream->Synchronize();
            }
            /*DrawRect*/ 
            rtlib::test::RenderFrameGL(ogl4Context, rectRendererGL.get(), frameBufferGL.get(), frameTextureGL.get());

            glfwContext->Update();

            isUpdated     = false;
            isResized     = false;
            isMovedCamera = false;
            {
                int tWidth, tHeight;
                glfwGetWindowSize(window->GetGLFWwindow(), &tWidth, &tHeight);
                if (width != tWidth || height != tHeight)
                {
                    std::cout << width <<  "->" << tWidth << "\n";
                    std::cout << height << "->" << tHeight << "\n";
                    width     = tWidth;
                    height    = tHeight;
                    isResized = true;
                    isUpdated = true;
                }
                else
                {
                    isResized = false;
                }
                float prevTime = glfwGetTime();
                {
                    windowState.delTime = windowState.curTime - prevTime;
                    windowState.curTime = prevTime;
                }
                isMovedCamera = rtlib::test::UpdateCameraMovement(
                    cameraController,
                    window.get(),
                    windowState.delTime,
                    windowState.delCurPos.x,
                    windowState.delCurPos.y
                );
                if (isMovedCamera) {
                    isUpdated = true;
                }
            }

            window->SwapBuffers();
        }
        {
            {
                sceneData.config.width = width ;
                sceneData.config.height= height;
                sceneJson = sceneData;
            }
            auto sceneJsonFile = std::ofstream(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", std::ios::binary);
            sceneJsonFile << sceneJson;
            sceneJsonFile.close();
        }
       
        {
            {
                for (auto& [name, texture] : textures) {
                    texture.Destroy();
                }
                textures.clear();
            }
            {
                for (auto& [name, geometryAS] : geometryASs) {
                    auto& [buffer, handle] = geometryAS;
                    buffer->Destroy();
                }
                geometryASs.clear();
            }
            {
                for (auto& [name, instBuffer] : instBuffers) {
                    instBuffer->Destroy();
                }
                instBuffers.clear();
            }
            topLevelAS.buffer->Destroy();
            paramsBuffer->Destroy();
            accumBufferCUDA->Destroy();
            frameBufferCUGL->Destroy();
            frameBufferGL->Destroy();
            stream->Destroy();
            window->Destroy();
            glfwContext->Terminate();
            opx7Context->Terminate();
        }
        window = nullptr;
    }
    catch (std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
    }
    return 0;
}
int main()
{
    OnlineSample();
}

