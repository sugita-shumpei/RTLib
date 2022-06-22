#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLibExtOPX7Test.h>
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

    int  width             = sceneJson.at("Config").at("Width").get<int>();
    int  height            = sceneJson.at("Config").at("Height").get<int>();
    int  samplesForLaunch  = sceneJson.at("Config").at("Samples").get<int>();
    int  maxSamples        = sceneJson.at("Config").at("MaxSamples").get<int>();
    int  samplesForAccum   = 0;
    auto cameraController  = sceneJson.at("CameraController").get<RTLib::Utils::CameraController>();
    auto objAssetLoader    = sceneJson.at("ObjModels").get<RTLib::Core::ObjModelAssetManager>();
    auto worldData = sceneJson.at("World").get<RTLib::Core::WorldData>();
    try
    {
        auto glfwContext = std::unique_ptr<GLFWContext>(GLFWContext::New());
        glfwContext->Initialize();

        auto window = std::unique_ptr<GL::GLFWOpenGLWindow>(rtlib::test::CreateGLFWWindow(glfwContext.get(), width, height, "title"));

        // contextはcopy/move不可
        auto contextCreateDesc           = OPX7ContextCreateDesc();
        contextCreateDesc.validationMode = OPX7ContextValidationMode::eALL;
        contextCreateDesc.level          = OPX7ContextValidationLogLevel::ePrint;
        auto opx7Context                 = std::make_unique<OPX7Context>(contextCreateDesc);
        auto ogl4Context                 = window->GetOpenGLContext();

        opx7Context->Initialize();

        auto shaderTableLayout = std::unique_ptr<OPX7ShaderTableLayout>();
        {
            auto blasLayouts = std::unordered_map<std::string, std::unique_ptr<OPX7ShaderTableLayoutGeometryAS>>();
            for (auto& [geometryASName, geometryAS] : worldData.geometryASs)
            {
                blasLayouts[geometryASName] = std::make_unique<OPX7ShaderTableLayoutGeometryAS>(geometryASName);
                auto buildInputSize = size_t(0);
                for (auto& geometryName : geometryAS.geometries)
                {
                    auto& objAsset = objAssetLoader.GetAsset(worldData.geometryObjModels[geometryName].base);
                    auto uniqueNames = objAsset.meshGroup->GetUniqueNames();
                    for (auto& uniqueName : uniqueNames)
                    {
                        auto mesh = objAsset.meshGroup->LoadMesh(uniqueName);
                        blasLayouts[geometryASName]->SetDwGeometry(OPX7ShaderTableLayoutGeometry(uniqueName, mesh->GetUniqueResource()->materials.size()));
                    }
                    buildInputSize += uniqueNames.size();
                }
            }

            auto tlasLayout = OPX7ShaderTableLayoutInstanceAS("Root");
            for (auto& instanceName : worldData.instanceASs["Root"].instances)
            {
                auto baseGeometryASName = worldData.instances[instanceName].base;
                tlasLayout.SetInstance(OPX7ShaderTableLayoutInstance(instanceName, blasLayouts[baseGeometryASName].get()));
            }
            tlasLayout.SetRecordStride(1);

            shaderTableLayout = std::make_unique<OPX7ShaderTableLayout>(tlasLayout);
        }

        auto accelBuildOptions = OptixAccelBuildOptions();
        {
            accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accelBuildOptions.motionOptions = {};
            accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        }
        auto blasHandles = std::unordered_map<std::string, OptixTraversableHandle>();
        auto blasBuffers = std::unordered_map<std::string, std::unique_ptr<CUDABuffer>>();

        {
            for (auto& [name,geometry  ]: worldData.geometryObjModels)
            {
                rtlib::test::InitMeshGroupExtData(opx7Context.get(), objAssetLoader.GetAsset(geometry.base).meshGroup);
            }
            for (auto &geometryASName: shaderTableLayout->GetBaseGeometryASNames())
            {
                auto& gasBaseDesc = shaderTableLayout->GetBaseDesc(geometryASName);
                {
                    auto* gasLayout = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS*>(gasBaseDesc.pData);
                    auto buildInputs = std::vector<OptixBuildInput>(gasLayout->GetDwGeometries().size());
                    auto vertexBufferPtrs = std::vector<CUdeviceptr>(gasLayout->GetDwGeometries().size());
                    auto geometryFlags    = std::vector<unsigned int>(gasLayout->GetDwGeometries().size());
                    auto buildInputOffset = size_t(0);
                    auto geometryAS       = worldData.geometryASs[geometryASName];
                    for (auto& geometryName : geometryAS.geometries)
                    {
                        auto &objAsset   = objAssetLoader.GetAsset(worldData.geometryObjModels[geometryName].base);
                        auto uniqueNames = objAsset.meshGroup->GetUniqueNames();
                        for (size_t i = 0; i < uniqueNames.size(); ++i)
                        {
                            auto mesh = objAsset.meshGroup->LoadMesh(uniqueNames[i]);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData *>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData *>(mesh->GetUniqueResource()->extData.get());
                            vertexBufferPtrs[buildInputOffset + i] = extSharedData->GetVertexBuffer();
                            geometryFlags[buildInputOffset + i]    = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                            buildInputs[buildInputOffset + i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                            {
                                buildInputs[buildInputOffset + i].triangleArray.vertexBuffers = &vertexBufferPtrs[buildInputOffset + i];
                                buildInputs[buildInputOffset + i].triangleArray.numVertices = extSharedData->GetVertexCount();
                                buildInputs[buildInputOffset + i].triangleArray.vertexFormat = extSharedData->GetVertexFormat();
                                buildInputs[buildInputOffset + i].triangleArray.vertexStrideInBytes = extSharedData->GetVertexStrideInBytes();
                                buildInputs[buildInputOffset + i].triangleArray.indexBuffer = extUniqueData->GetTriIdxBuffer();
                                buildInputs[buildInputOffset + i].triangleArray.numIndexTriplets = extUniqueData->GetTriIdxCount();
                                buildInputs[buildInputOffset + i].triangleArray.indexFormat = extUniqueData->GetTriIdxFormat();
                                buildInputs[buildInputOffset + i].triangleArray.indexStrideInBytes = extUniqueData->GetTriIdxStrideInBytes();
                                buildInputs[buildInputOffset + i].triangleArray.sbtIndexOffsetBuffer = extUniqueData->GetMatIdxBuffer();
                                buildInputs[buildInputOffset + i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
                                buildInputs[buildInputOffset + i].triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
                                buildInputs[buildInputOffset + i].triangleArray.numSbtRecords = shaderTableLayout->GetBaseDesc(geometryASName +"/"+uniqueNames[i]).baseRecordCount;
                                buildInputs[buildInputOffset + i].triangleArray.preTransform = 0;
                                buildInputs[buildInputOffset + i].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
                                buildInputs[buildInputOffset + i].triangleArray.flags = &geometryFlags[buildInputOffset + i];
                            }
                        }
                        buildInputOffset += uniqueNames.size();
                    }
                    auto accelOutput = OPX7Natives::BuildAccelerationStructure(opx7Context.get(), accelBuildOptions, buildInputs);
                    blasHandles[geometryASName] = accelOutput.handle;
                    blasBuffers[geometryASName] = std::move(accelOutput.buffer);
                }
            }
        }


        auto instBuffers = std::unordered_map<std::string, std::unique_ptr<CUDABuffer>>();
        auto tlasHandle  = OptixTraversableHandle();
        auto tlasBuffer  = std::unique_ptr<CUDABuffer>();
        
        {
            auto instIndices = std::unordered_map<std::string, unsigned int>();
            {
                unsigned int i = 0;
                for (auto& [name, instance] : worldData.instances) {
                    instIndices[name] = i;
                    ++i;
                }
            }
            auto opx7Instances = std::vector<OptixInstance>();
            {
                opx7Instances.reserve(worldData.instanceASs["Root"].instances.size());
                for (auto& instanceName : worldData.instanceASs["Root"].instances)
                {
                    auto opx7Instance = OptixInstance();
                    auto transforms = worldData.instances[instanceName].transform;
                    opx7Instance.traversableHandle = blasHandles[worldData.instances[instanceName].base];
                    opx7Instance.visibilityMask = 255;
                    opx7Instance.instanceId = instIndices[instanceName];
                    opx7Instance.sbtOffset = shaderTableLayout->GetDesc("Root/" + instanceName).recordOffset;
                    opx7Instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                    std::memcpy(opx7Instance.transform, transforms.data(), transforms.size() * sizeof(float));
                    opx7Instances.push_back(opx7Instance);
                }


            }

            instBuffers["Root"] = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer(
                CUDABufferCreateDesc{
                    CUDAMemoryFlags::eDefault,
                    sizeof(OptixInstance) * opx7Instances.size(),
                    opx7Instances.data() }));

            auto buildInputs = std::vector<OptixBuildInput>(1);
            buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            buildInputs[0].instanceArray.instances = CUDANatives::GetCUdeviceptr(instBuffers["Root"].get());
            buildInputs[0].instanceArray.numInstances = opx7Instances.size();

            auto accelOutput = OPX7Natives::BuildAccelerationStructure(opx7Context.get(), accelBuildOptions, buildInputs);
            tlasHandle = accelOutput.handle;
            tlasBuffer = std::move(accelOutput.buffer);
        }

        auto pipelineCompileOptions = OPX7PipelineCompileOptions{};
        {
            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPX7TraversableGraphFlagsAllowSingleLevelInstancing;
            pipelineCompileOptions.numAttributeValues = 3;
            pipelineCompileOptions.numPayloadValues = 3;
            pipelineCompileOptions.launchParamsVariableNames = "params";
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPX7PrimitiveTypeFlagsTriangle;
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
        auto raygPG = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Raygen({module.get(), "__raygen__rg"})));
        auto missPG = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Miss({module.get(), "__miss__ms"})));
        auto hitgPG = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Hitgroup({module.get(), "__closesthit__ch"})));
        auto pipelineCreateDesc = OPX7PipelineCreateDesc{};
        {
            pipelineCreateDesc.linkOptions.maxTraceDepth = 1;
            pipelineCreateDesc.linkOptions.debugLevel = OPX7CompileDebugLevel::eMinimal;
            pipelineCreateDesc.compileOptions = pipelineCompileOptions;
            pipelineCreateDesc.programGroups = {raygPG.get(), missPG.get(), hitgPG.get()};
        }
        auto pipeline = std::unique_ptr<OPX7Pipeline>(opx7Context->CreateOPXPipeline(pipelineCreateDesc));

        auto shaderTableDesc = OPX7ShaderTableCreateDesc();
        {

            shaderTableDesc.raygenRecordSizeInBytes     = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
            shaderTableDesc.missRecordStrideInBytes     = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
            shaderTableDesc.missRecordCount             = shaderTableLayout->GetRecordStride();
            shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
            shaderTableDesc.hitgroupRecordCount         = shaderTableLayout->GetRecordCount();
        }
        auto shaderTable = std::unique_ptr<OPX7ShaderTable>(opx7Context->CreateOPXShaderTable(shaderTableDesc));

        {
            {
                auto raygenRecord     = raygPG->GetRecord<RayGenData>();
                auto camera           = cameraController.GetCamera(static_cast<float>(width) / static_cast<float>(height));
                raygenRecord.data.eye = camera.GetEyeAs<float3>();
                auto [u, v, w]        = camera.GetUVW();
                raygenRecord.data.u   = make_float3(u[0], u[1], u[2]);
                raygenRecord.data.v   = make_float3(v[0], v[1], v[2]);
                raygenRecord.data.w   = make_float3(w[0], w[1], w[2]);
                shaderTable->SetHostRaygenRecordTypeData(raygenRecord);
            }
            {
                auto missRecord = missPG->GetRecord<MissData>();
                missRecord.data.bgColor = float4{1.0f, 1.0f, 1.0f, 1.0f};
                shaderTable->SetHostMissRecordTypeData(0, missRecord);
            }
            auto sbtStride = shaderTableLayout->GetRecordStride();
            for (auto &instanceName : worldData.instanceASs["Root"].instances)
            {
                if (worldData.instances.at(instanceName).asType == "Geometry")
                {
                    auto   baseGasName = worldData.instances.at(instanceName).base;
                    for (auto &geometryName : worldData.geometryASs[baseGasName].geometries)
                    {
                        auto objAsset = objAssetLoader.GetAsset(worldData.geometryObjModels[geometryName].base);
                        for (auto& uniqueName : objAsset.meshGroup->GetUniqueNames()) {
                            auto mesh = objAsset.meshGroup->LoadMesh(uniqueName);
                            auto extSharedData = mesh->GetSharedResource()->GetExtData<rtlib::test::OPX7MeshSharedResourceExtData>();
                            auto extUniqueData = mesh->GetUniqueResource()->GetExtData<rtlib::test::OPX7MeshUniqueResourceExtData>();
                            auto desc = shaderTableLayout->GetDesc("Root/" + instanceName + "/" + uniqueName);
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupRecord = hitgPG->GetRecord<HitgroupData>();
                                {
                                    auto matIdx = i / desc.recordStride;
                                    auto material = objAsset.materials[mesh->GetUniqueResource()->materials[matIdx]];
                                    hitgroupRecord.data.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBuffer());
                                    hitgroupRecord.data.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBuffer());
                                    hitgroupRecord.data.diffuse  = material.GetFloat3As<float3>("diffCol");
                                    hitgroupRecord.data.emission = material.GetFloat3As<float3>("emitCol");
                                }
                                shaderTable->SetHostHitgroupRecordTypeData(desc.recordOffset + i, hitgroupRecord);
                            }
                        }
                        
                    }
                }
            }
            shaderTable->Upload();
        }
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
        auto frameBufferGL   = std::unique_ptr<GLBuffer>(ogl4Context->CreateBuffer(GLBufferCreateDesc{sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr}));
        auto frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
        auto frameTextureGL  = std::unique_ptr<GLTexture>();
        auto frameTextureDesc = RTLib::Ext::GL::GLTextureCreateDesc();
        {
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
        auto rectRendererGL = std::unique_ptr<GLRectRenderer>(ogl4Context->CreateRectRenderer({1, false, true}));

        auto paramsBuffer = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer({CUDAMemoryFlags::eDefault, static_cast<size_t>(sizeof(Params)), nullptr}));
        auto stream = std::unique_ptr<CUDAStream>(opx7Context->CreateStream());

        auto isUpdated = false;
        auto isResized = false;
        auto isMovedCamera = false;

        auto windowState = WindowState();
        window->Show();
        window->SetResizable(true);
        window->SetUserPointer(&windowState);
        window->SetCursorPosCallback(cursorPosCallback);
        while (!window->ShouldClose())
        {
            if (samplesForAccum >= maxSamples) {
                break;
            }
            {
                if (isResized)
                {
                    frameBufferCUGL->Destroy();
                    frameBufferGL->Destroy();
                    frameTextureGL->Destroy();
                    frameBufferGL   = std::unique_ptr<GLBuffer>(ogl4Context->CreateBuffer(GLBufferCreateDesc{sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr}));
                    frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
                    frameTextureDesc.image.extent = {(uint32_t)width, (uint32_t)height, (uint32_t)0};
                    frameTextureGL  = std::unique_ptr<GLTexture>(ogl4Context->CreateTexture(frameTextureDesc));
                }
                if (isUpdated) {
                    auto zeroClearData = std::vector<float>(width * height * 3, 0.0f);
                    opx7Context->CopyMemoryToBuffer(
                        accumBufferCUDA.get(), {{zeroClearData.data(), 0, sizeof(zeroClearData[0])* zeroClearData.size()}}
                    );
                    samplesForAccum = 0;
                }
                if (isMovedCamera)
                {
                    auto raygenRecord = raygPG->GetRecord<RayGenData>();
                    {
                        auto camera = cameraController.GetCamera(static_cast<float>(width) / static_cast<float>(height));
                        raygenRecord.data.eye = camera.GetEyeAs<float3>();
                        auto [u, v, w] = camera.GetUVW();
                        raygenRecord.data.u = make_float3(u[0], u[1], u[2]);
                        raygenRecord.data.v = make_float3(v[0], v[1], v[2]);
                        raygenRecord.data.w = make_float3(w[0], w[1], w[2]);
                    }
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
                        params.accumBuffer = reinterpret_cast<float3*>(CUDANatives::GetCUdeviceptr(accumBufferCUDA.get()));
                        params.seedBuffer  = reinterpret_cast<unsigned int*>(CUDANatives::GetCUdeviceptr( seedBufferCUDA.get()));
                        params.frameBuffer = reinterpret_cast<uchar4 *>(CUDANatives::GetCUdeviceptr(frameBufferCUDA));
                        params.width       = width;
                        params.height      = height;
                        params.samplesForAccum  = samplesForAccum;
                        params.samplesForLaunch = samplesForLaunch;
                        params.gasHandle = tlasHandle;
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
                    windowState.delCurPos.y);
                if (isMovedCamera) {
                    isUpdated = true;
                }
            }

            window->SwapBuffers();
        }
        {
            {
                sceneJson["CameraController"] = cameraController;
                sceneJson["Width"]            = width;
                sceneJson["Height"]           = height;
            }
            auto sceneJsonFile = std::ofstream(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", std::ios::binary);
            sceneJsonFile << sceneJson;
            sceneJsonFile.close();
        }
        {
            for (auto& [name, blasBuffer] : blasBuffers) {
                blasBuffer->Destroy();
            }
            blasBuffers.clear();
        }
        {
            for (auto& [name, instBuffer] : instBuffers) {
                instBuffer->Destroy();
            }
            instBuffers.clear();
        }
        tlasBuffer->Destroy();
        paramsBuffer->Destroy();
        accumBufferCUDA->Destroy();
        frameBufferCUGL->Destroy();
        frameBufferGL->Destroy();
        stream->Destroy();
        window->Destroy();
        glfwContext->Terminate();
        opx7Context->Terminate();
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
