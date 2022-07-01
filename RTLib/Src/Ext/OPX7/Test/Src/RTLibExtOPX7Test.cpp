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
    using namespace RTLib::Ext::GL  ;
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

        auto cudaTextures = sceneData.LoadCudaTextures(opx7Context.get());
        auto geometryASs  = sceneData.BuildGeometryASs(opx7Context.get(), accelBuildOptions);
        auto instanceASs  = sceneData.BuildInstanceASs(opx7Context.get(), accelBuildOptions, shaderTableLayout.get(), geometryASs);

        auto pipelineCompileOptions = OPX7PipelineCompileOptions{};
        {
            pipelineCompileOptions.usesMotionBlur            = false;
            pipelineCompileOptions.traversableGraphFlags     = 0;
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
        auto missPGs = std::vector<std::unique_ptr<OPX7ProgramGroup>>(RAY_TYPE_COUNT);
        missPGs[RAY_TYPE_RADIANCE] = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Miss({ module.get(), "__miss__radiance" })));
        missPGs[RAY_TYPE_OCCLUDED] = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Miss({ module.get(), "__miss__occluded" })));
        auto hitgPGs = std::vector<std::unique_ptr<OPX7ProgramGroup>>(RAY_TYPE_COUNT);
        hitgPGs[RAY_TYPE_RADIANCE] = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Hitgroup({module.get(), "__closesthit__radiance"})));
        hitgPGs[RAY_TYPE_OCCLUDED] = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Hitgroup({ module.get(),"__closesthit__occluded" })));
        auto excpPG = std::unique_ptr<OPX7ProgramGroup>(opx7Context->CreateOPXProgramGroup(OPX7ProgramGroupCreateDesc::Exception({ module.get(), "__exception__ep" })));
        auto pipelineCreateDesc = OPX7PipelineCreateDesc{};
        {
            pipelineCreateDesc.linkOptions.maxTraceDepth = 1;
            pipelineCreateDesc.linkOptions.debugLevel    = OPX7CompileDebugLevel::eMinimal;
            pipelineCreateDesc.compileOptions = pipelineCompileOptions;
            pipelineCreateDesc.programGroups  = {
                raygPG.get(), 
                missPGs[RAY_TYPE_RADIANCE].get(),
                missPGs[RAY_TYPE_OCCLUDED].get(),
                hitgPGs[RAY_TYPE_RADIANCE].get(),
                hitgPGs[RAY_TYPE_OCCLUDED].get(),
                excpPG.get()
            };
        }
        auto pipeline = std::unique_ptr<OPX7Pipeline>(opx7Context->CreateOPXPipeline(pipelineCreateDesc));
        {
            auto cssRG = raygPG->GetStackSize().cssRG;
            auto cssMS = std::max(missPGs[RAY_TYPE_RADIANCE]->GetStackSize().cssMS, missPGs[RAY_TYPE_OCCLUDED]->GetStackSize().cssMS);
            auto cssCH = std::max(hitgPGs[RAY_TYPE_RADIANCE]->GetStackSize().cssCH, hitgPGs[RAY_TYPE_OCCLUDED]->GetStackSize().cssCH);
            auto cssIS = std::max(hitgPGs[RAY_TYPE_RADIANCE]->GetStackSize().cssIS, hitgPGs[RAY_TYPE_OCCLUDED]->GetStackSize().cssIS);
            auto cssAH = std::max(hitgPGs[RAY_TYPE_RADIANCE]->GetStackSize().cssAH, hitgPGs[RAY_TYPE_OCCLUDED]->GetStackSize().cssAH);
            auto cssCCTree = 0;
            auto cssCHOrMsPlusCCTree = std::max(cssMS, cssCH) + cssCCTree;
            auto continuationStackSizes = cssRG + cssCCTree +
                (std::max<unsigned int>(1, pipeline->GetLinkOptions().maxTraceDepth) - 1) * cssCHOrMsPlusCCTree +
                (std::min<unsigned int>(1, pipeline->GetLinkOptions().maxTraceDepth)) * std::max(cssCHOrMsPlusCCTree, cssIS + cssAH);
            pipeline->SetStackSize(0, 0, continuationStackSizes, shaderTableLayout->GetMaxTraversableDepth());
        }

        auto shaderTable = std::unique_ptr<OPX7ShaderTable>();
        {

            auto shaderTableDesc = OPX7ShaderTableCreateDesc();
            {

                shaderTableDesc.raygenRecordSizeInBytes     = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
                shaderTableDesc.missRecordStrideInBytes     = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
                shaderTableDesc.missRecordCount             = shaderTableLayout->GetRecordStride();
                shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
                shaderTableDesc.hitgroupRecordCount         = shaderTableLayout->GetRecordCount();
                shaderTableDesc.exceptionRecordSizeInBytes  = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<unsigned int>);
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
                shaderTable->SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, missPGs[RAY_TYPE_RADIANCE]->GetRecord<MissData>({ {} }));
                shaderTable->SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, missPGs[RAY_TYPE_OCCLUDED]->GetRecord<MissData>({ }));
            }
            {
                shaderTable->SetHostExceptionRecordTypeData(excpPG->GetRecord<unsigned int>());
            }
            for (auto &instanceName : shaderTableLayout->GetInstanceNames())
            {
                auto& instanceDesc = shaderTableLayout->GetDesc(instanceName);
                auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
                auto  geometryAS   = instanceData->GetDwGeometryAS();
                if   (geometryAS)
                {
                    for (auto &geometryName : worldData.geometryASs[geometryAS->GetName()].geometries)
                    {
                        auto& geometry = worldData.geometryObjModels[geometryName];
                        auto objAsset  = objAssetLoader.GetAsset(geometry.base);
                        for (auto& [meshName,meshData] : geometry.meshes) {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshName);
                            auto desc = shaderTableLayout->GetDesc(instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupRecord = hitgPGs[i%desc.recordStride]->GetRecord<HitgroupData>();
                                if ((i%desc.recordStride)==RAY_TYPE_RADIANCE) {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupRecord.data.vertices = reinterpret_cast<float3*>(CUDANatives::GetCUdeviceptr(extSharedData->GetVertexBufferView()));
                                    hitgroupRecord.data.indices  = reinterpret_cast< uint3*>(CUDANatives::GetCUdeviceptr(extUniqueData->GetTriIdxBufferView()));
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
                                    hitgroupRecord.data.diffuseTex  = RTLib::Ext::CUDA::CUDANatives::GetCUtexObject(cudaTextures.at(diffTexStr).handle.get());
                                    hitgroupRecord.data.specularTex = RTLib::Ext::CUDA::CUDANatives::GetCUtexObject(cudaTextures.at(specTexStr).handle.get());
                                    hitgroupRecord.data.emissionTex = RTLib::Ext::CUDA::CUDANatives::GetCUtexObject(cudaTextures.at(emitTexStr).handle.get());
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
            auto mt19937  = std::mt19937();
            auto seedData = std::vector<unsigned int>(width * height * sizeof(unsigned int));
            std::generate(std::begin(seedData), std::end(seedData), mt19937);
            seedBufferCUDA = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer(
                {CUDAMemoryFlags::eDefault,seedData.size() * sizeof(seedData[0]),seedData.data()}
            ));
        }

        auto accumBufferCUDA = std::unique_ptr<CUDABuffer>(opx7Context->CreateBuffer(  { CUDAMemoryFlags::eDefault,width* height * sizeof(float)*3,nullptr }  ));
        auto frameBufferGL   = std::unique_ptr<GLBuffer  >(ogl4Context->CreateBuffer(GLBufferCreateDesc{sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr}));
        auto frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
        auto frameTextureGL  = rtlib::test::CreateFrameTextureGL(ogl4Context, width, height);
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
                    frameTextureGL  = rtlib::test::CreateFrameTextureGL(ogl4Context, width, height);
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
                        params.gasHandle        = instanceASs["Root"].handle;
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
                for (auto& [name, texture] : cudaTextures) {
                    texture.Destroy();
                }
                cudaTextures.clear();
            }
            {
                for (auto& [name, geometryAS] : geometryASs) {
                    auto& [buffer, handle] = geometryAS;
                    buffer->Destroy();
                }
                geometryASs.clear();
            }
            {
                for (auto& [name, instanceAS] : instanceASs) {
                    auto& [buffer, handle, instanceBuffer,instanceArray] = instanceAS;
                    instanceBuffer->Destroy();
                    buffer->Destroy();
                }
                instanceASs.clear();
            }
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

