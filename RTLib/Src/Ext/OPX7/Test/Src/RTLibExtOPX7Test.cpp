#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLibExtOPX7Test.h>
struct WindowState
{
    float curTime = 0.0f;
    float delTime = 0.0f;
    float2 curCurPos = {};
    float2 delCurPos = {};
};

int OnlineSample()
{
    using namespace RTLib::Ext::CUDA;
    using namespace RTLib::Ext::OPX7;
    using namespace RTLib::Ext::GL;
    using namespace RTLib::Ext::CUGL;
    using namespace RTLib::Ext::GLFW;

    auto sceneJson = nlohmann::json();
    {
        std::ifstream sceneJsonFile("./scene.json", std::ios::binary);
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
    auto cameraController = sceneJson.at("CameraController").get<RTLib::Utils::CameraController>();
    auto objAssetLoader = RTLib::Core::ObjModelAssetManager(sceneJson.at("ObjModels").at("CacheDir").get<std::string>());
    for (auto &assetJson : sceneJson.at("ObjModels").at("Assets").items())
    {
        objAssetLoader.LoadAsset(assetJson.key(), assetJson.value().get<std::string>());
    }
    // auto vertices = box.getVertices();
    // auto indices = box.getIndices();
    try
    {
        int width = sceneJson.at("Width").get<int>();
        int height = sceneJson.at("Height").get<int>();
        auto glfwContext = std::unique_ptr<GLFWContext>(GLFWContext::New());
        glfwContext->Initialize();

        auto window = std::unique_ptr<GL::GLFWOpenGLWindow>(rtlib::test::CreateGLFWWindow(glfwContext.get(), width, height, "title"));

        // contextはcopy/move不可
        auto contextCreateDesc = OPX7ContextCreateDesc();
        contextCreateDesc.validationMode = OPX7ContextValidationMode::eALL;
        contextCreateDesc.level = OPX7ContextValidationLogLevel::ePrint;
        auto opx7Context = std::make_unique<OPX7Context>(contextCreateDesc);
        auto ogl4Context = window->GetOpenGLContext();

        opx7Context->Initialize();
        for (auto &geometry : sceneJson.at("World").at("Geometries").items())
        {
            if (geometry.value().at("Type").get<std::string>() == "ObjModel")
            {
                auto objAssetKey = geometry.value().at("Base").get<std::string>();
                auto &objAsset = objAssetLoader.GetAsset(objAssetKey);
                {
                    auto sharedResource = objAsset.meshGroup->GetSharedResource();
                    sharedResource->AddExtData<rtlib::test::OPX7MeshSharedResourceExtData>(opx7Context.get());
                    auto extData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData *>(sharedResource->extData.get());
                    extData->SetVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
                    extData->SetVertexStrideInBytes(sizeof(float) * 3);
                }
                for (auto &[name, uniqueResource] : objAsset.meshGroup->GetUniqueResources())
                {
                    uniqueResource->AddExtData<rtlib::test::OPX7MeshUniqueResourceExtData>(opx7Context.get());
                    auto extData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData *>(uniqueResource->extData.get());
                    extData->SetTriIdxFormat(OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
                    extData->SetTriIdxStrideInBytes(sizeof(uint32_t) * 3);
                }
            }
        }
        auto accelBuildOptions = OptixAccelBuildOptions();
        accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelBuildOptions.motionOptions = {};
        accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        auto blasLayouts = std::unordered_map<std::string, std::unique_ptr<OPX7ShaderTableLayoutGeometryAS>>();
        auto blasHandles = std::unordered_map<std::string, OptixTraversableHandle>();
        auto blasBuffers = std::unordered_map<std::string, std::unique_ptr<CUDABuffer>>();
        {
            auto geometryFlags = std::vector<unsigned int>();
            for (auto &geometryASJson : sceneJson.at("World").at("GeometryASs").items())
            {
                blasLayouts[geometryASJson.key()] = std::make_unique<OPX7ShaderTableLayoutGeometryAS>(geometryASJson.key());
                auto buildInputs = std::vector<OptixBuildInput>();
                auto d_Vertices = std::vector<CUdeviceptr>();
                auto geometryNames = geometryASJson.value().at("Geometries").get<std::vector<std::string>>();
                auto &objAsset = objAssetLoader.GetAsset("Bistro-Exterior");
                auto uniqueNames = objAsset.meshGroup->GetUniqueNames();
                d_Vertices.resize(uniqueNames.size());
                buildInputs.resize(uniqueNames.size());
                geometryFlags.resize(uniqueNames.size());
                for (size_t i = 0; i < buildInputs.size(); ++i)
                {
                    auto geometryASName = geometryASJson.key();
                    auto mesh = objAsset.meshGroup->LoadMesh(uniqueNames[i]);
                    auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData *>(mesh->GetSharedResource()->extData.get());
                    auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData *>(mesh->GetUniqueResource()->extData.get());
                    d_Vertices[i] = extSharedData->GetVertexBuffer();
                    geometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                    buildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                    blasLayouts[geometryASName]->SetDwGeometry(OPX7ShaderTableLayoutGeometry(uniqueNames[i], mesh->GetUniqueResource()->materials.size()));
                    {
                        buildInputs[i].triangleArray.vertexBuffers = &d_Vertices[i];
                        buildInputs[i].triangleArray.numVertices = extSharedData->GetVertexCount();
                        buildInputs[i].triangleArray.vertexFormat = extSharedData->GetVertexFormat();
                        buildInputs[i].triangleArray.vertexStrideInBytes = extSharedData->GetVertexStrideInBytes();
                        buildInputs[i].triangleArray.indexBuffer = extUniqueData->GetTriIdxBuffer();
                        buildInputs[i].triangleArray.numIndexTriplets = extUniqueData->GetTriIdxCount();
                        buildInputs[i].triangleArray.indexFormat = extUniqueData->GetTriIdxFormat();
                        buildInputs[i].triangleArray.indexStrideInBytes = extUniqueData->GetTriIdxStrideInBytes();
                        buildInputs[i].triangleArray.sbtIndexOffsetBuffer = extUniqueData->GetMatIdxBuffer();
                        buildInputs[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
                        buildInputs[i].triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
                        buildInputs[i].triangleArray.numSbtRecords = blasLayouts[geometryASName]->GetDwGeometries().back().GetBaseRecordCount();
                        buildInputs[i].triangleArray.preTransform = 0;
                        buildInputs[i].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
                        buildInputs[i].triangleArray.flags = &geometryFlags[i];
                    }
                }
                auto accelOutput = OPX7Natives::BuildAccelerationStructure(opx7Context.get(), accelBuildOptions, buildInputs);
                blasHandles[geometryASJson.key()] = accelOutput.handle;
                blasBuffers[geometryASJson.key()] = std::move(accelOutput.buffer);
            }
        }
        auto blasInstBuffer = std::unique_ptr<CUDABuffer>();
        auto tlasHandle = OptixTraversableHandle();
        auto tlasLayout = OPX7ShaderTableLayoutInstanceAS("Root");
        auto tlasBuffer = std::unique_ptr<CUDABuffer>();
        /*    {
                auto  instance = OptixInstance();
                float transforms[12] = {
                    1.0f,0.0f,0.0f,0.0f,
                    0.0f,1.0f,0.0f,0.0f,
                    0.0f,0.0f,1.0f,0.0f
                };
                std::memcpy(instance.transform, transforms, sizeof(float) * 12);
                instance.traversableHandle = blasHandle;
                instance.visibilityMask    = OptixVisibilityMask(255);
                instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
                instance.sbtOffset         = 0;
                instance.instanceId        = 0;

                opx7Context->CopyMemoryToBuffer(blasInstBuffer.get(), { {&instance,(size_t)0,(size_t)sizeof(instance)}});

                auto buildInputs = std::vector<OptixBuildInput>(1);
                buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
                buildInputs[0].instanceArray.instances    = CUDANatives::GetCUdeviceptr(blasInstBuffer.get());
                buildInputs[0].instanceArray.numInstances = 1;

                auto accelOutput = OPX7Natives::BuildAccelerationStructure(opx7Context.get(), accelBuildOptions, buildInputs);
                tlasHandle = accelOutput.handle;
                tlasBuffer = std::move(accelOutput.buffer);


            }*/
        {
            auto rootInstanceNames = sceneJson.at("World").at("InstanceASs").at("Root").at("Instances").get<std::vector<std::string>>();
            for (auto &instanceName : rootInstanceNames)
            {
                auto baseGeometryASName = sceneJson.at("World").at("Instances").at(instanceName).at("Base").get<std::string>();
                tlasLayout.SetInstance(OPX7ShaderTableLayoutInstance("Instance0", blasLayouts[baseGeometryASName].get()));
            }

            tlasLayout.SetRecordStride(1);
        }
        auto pipelineCompileOptions = OPX7PipelineCompileOptions{};
        {
            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPX7TraversableGraphFlagsAllowSingleGAS;
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
        auto raygenPG = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Raygen({module.get(), "__raygen__rg"})));
        auto missPG = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Miss({module.get(), "__miss__ms"})));
        auto hitgroupPG = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Hitgroup({module.get(), "__closesthit__ch"})));
        auto pipelineCreateDesc = OPX7PipelineCreateDesc{};
        {
            pipelineCreateDesc.linkOptions.maxTraceDepth = 1;
            pipelineCreateDesc.linkOptions.debugLevel = OPX7CompileDebugLevel::eMinimal;
            pipelineCreateDesc.compileOptions = pipelineCompileOptions;
            pipelineCreateDesc.programGroups = {raygenPG.get(), missPG.get(), hitgroupPG.get()};
        }
        auto pipeline = std::unique_ptr<OPX7Pipeline>(opx7Context->CreateOPXPipeline(pipelineCreateDesc));

        auto shaderTableLayout = OPX7ShaderTableLayout(tlasLayout);
        auto shaderTableDesc = OPX7ShaderTableCreateDesc();
        {

            shaderTableDesc.raygenRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
            shaderTableDesc.missRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
            shaderTableDesc.missRecordCount = shaderTableLayout.GetRecordStride();
            shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
            shaderTableDesc.hitgroupRecordCount = shaderTableLayout.GetRecordCount();
        }
        auto shaderTable = std::unique_ptr<OPX7ShaderTable>(opx7Context->CreateOPXShaderTable(shaderTableDesc));

        {
            {
                auto raygenRecord = raygenPG->GetRecord<RayGenData>();
                auto camera = cameraController.GetCamera(static_cast<float>(width) / static_cast<float>(height));
                raygenRecord.data.eye = camera.GetEyeAs<float3>();
                auto [u, v, w] = camera.GetUVW();
                raygenRecord.data.u = make_float3(u[0], u[1], u[2]);
                raygenRecord.data.v = make_float3(v[0], v[1], v[2]);
                raygenRecord.data.w = make_float3(w[0], w[1], w[2]);
                shaderTable->SetHostRaygenRecordTypeData(raygenRecord);
            }
            {
                auto missRecord = missPG->GetRecord<MissData>();
                missRecord.data.bgColor = float4{1.0f, 1.0f, 1.0f, 1.0f};
                shaderTable->SetHostMissRecordTypeData(0, missRecord);
            }
            {
                auto &objAsset = objAssetLoader.GetAsset("Bistro-Exterior");
                auto instLayout = shaderTableLayout.FindInstance("Instance0");
                auto gasLayout = shaderTableLayout.FindGeometryAS(instLayout, "Bistro-Exterior");

                auto sbtStride = shaderTableLayout.GetRecordStride();
                auto sbtOffset = instLayout->GetRecordOffset();

                for (auto &geometryLayout : gasLayout->GetDwGeometries())
                {
                    auto mesh = objAsset.meshGroup->LoadMesh(geometryLayout.GetName());
                    auto extSharedData = mesh->GetSharedResource()->GetExtData<rtlib::test::OPX7MeshSharedResourceExtData>();
                    auto extUniqueData = mesh->GetUniqueResource()->GetExtData<rtlib::test::OPX7MeshUniqueResourceExtData>();
                    auto hitgroupRecord = hitgroupPG->GetRecord<HitgroupData>();
                    hitgroupRecord.data.vertices = reinterpret_cast<float3 *>(extSharedData->GetVertexBuffer());
                    hitgroupRecord.data.indices = reinterpret_cast<uint3 *>(extUniqueData->GetTriIdxBuffer());
                    hitgroupRecord.data.diffuse = make_float3(1.0f, 1.0f, 1.0f);
                    hitgroupRecord.data.emission = make_float3(0.3f, 0.3f, 0.3f);
                    for (auto i = 0; i < geometryLayout.GetBaseRecordCount(); ++i)
                    {
                        for (auto j = 0; j < sbtStride; ++j)
                        {
                            shaderTable->SetHostHitgroupRecordTypeData(
                                sbtOffset + sbtStride * (geometryLayout.GetBaseRecordOffset() + i) + j,
                                hitgroupRecord);
                        }
                    }
                }
                auto pCpuHitgroupRecord = shaderTable->GetHostHitgroupRecordTypeData<HitgroupData>(0);
                auto cpuHitgroupRecordArray = std::vector<RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>>(
                    pCpuHitgroupRecord, pCpuHitgroupRecord + shaderTable->GetHitgroupRecordCount());
            }
            shaderTable->Upload();
        }

        auto frameBufferGL = std::unique_ptr<GLBuffer>(ogl4Context->CreateBuffer(GLBufferCreateDesc{sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr}));
        auto frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
        auto frameTextureGL = std::unique_ptr<GLTexture>();
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

        auto dPixels = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer({CUDAMemoryFlags::eDefault, static_cast<size_t>(sizeof(uchar4) * width * height), nullptr}));
        auto dParams = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer({CUDAMemoryFlags::eDefault, static_cast<size_t>(sizeof(Params)), nullptr}));
        auto stream = std::unique_ptr<CUDAStream>(opx7Context->CreateStream());

        auto isUpdated = false;
        auto isResized = false;
        auto isMovedCamera = false;
        auto windowState = WindowState();
        window->Show();
        window->SetResizable(true);
        while (!window->ShouldClose())
        {
            {
                if (isResized)
                {
                    frameBufferCUGL->Destroy();
                    frameBufferGL->Destroy();
                    frameTextureGL->Destroy();
                    frameBufferGL = std::unique_ptr<GLBuffer>(ogl4Context->CreateBuffer(GLBufferCreateDesc{sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr}));
                    frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
                    frameTextureDesc.image.extent = {(uint32_t)width, (uint32_t)height, (uint32_t)0};
                    frameTextureGL = std::unique_ptr<GLTexture>(ogl4Context->CreateTexture(frameTextureDesc));
                }
                if (isMovedCamera)
                {
                    auto raygenRecord = raygenPG->GetRecord<RayGenData>();
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
                        params.image = reinterpret_cast<uchar4 *>(CUDANatives::GetCUdeviceptr(frameBufferCUDA));
                        params.width = width;
                        params.height = height;
                        params.gasHandle = blasHandles["Bistro-Exterior"];
                    }
                    stream->CopyMemoryToBuffer(dParams.get(), {{&params, 0, sizeof(params)}});
                    pipeline->Launch(stream.get(), CUDABufferView(dParams.get(), 0, dParams->GetSizeInBytes()), shaderTable.get(), width, height, 1);
                }
                frameBufferCUGL->Unmap(stream.get());
                stream->Synchronize();
            }
            /*DrawRect*/ {
                ogl4Context->SetClearBuffer(GLClearBufferFlagsColor);
                ogl4Context->SetClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                ogl4Context->CopyBufferToImage(frameBufferGL.get(), frameTextureGL->GetImage(), {GLBufferImageCopy{
                                                                                                    ((size_t)0),
                                                                                                    ((size_t)0),
                                                                                                    ((size_t)0),
                                                                                                    {((uint32_t)0), ((uint32_t)0), ((uint32_t)1)},
                                                                                                    {0, 0, 0},
                                                                                                    {((uint32_t)width), ((uint32_t)height), ((uint32_t)1)},
                                                                                                }});
                rectRendererGL->DrawTexture(frameTextureGL.get());
            }
            glfwContext->Update();
            isUpdated = false;
            isResized = false;
            isMovedCamera = false;
            {
                int tWidth, tHeight;
                glfwGetWindowSize(window->GetGLFWwindow(), &tWidth, &tHeight);
                if (width != tWidth || height != tHeight)
                {
                    std::cout << width  << "->" << tWidth  << "\n";
                    std::cout << height << "->" << tHeight << "\n";
                    width  = tWidth;
                    height = tHeight;
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
            }

            window->SwapBuffers();
        }

        {
            {
                sceneJson["CameraController"] = cameraController;
                sceneJson["Width"] = width;
                sceneJson["Height"] = height;
            }
            auto sceneJsonFile = std::ofstream("./scene.json", std::ios::binary);
            sceneJsonFile << sceneJson;
            sceneJsonFile.close();
        }
        frameBufferCUGL->Destroy();
        frameBufferGL->Destroy();
        stream->Destroy();
        window->Destroy();
        glfwContext->Terminate();
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
