#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLibExtOPX7Test.h>


struct  EventState
{
    bool isUpdated = false;
    bool isResized = false;
    bool isMovedCamera = false;

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
class RTLibExtOPX7TestApplication
{
public :
    RTLibExtOPX7TestApplication(const std::string& scenePath)noexcept
    {
        m_ScenePath = scenePath;
    }
    void LoadScene() {
        m_SceneData = rtlib::test::LoadScene(m_ScenePath);
    }
    void SaveScene() {
        rtlib::test::SaveScene(m_ScenePath, m_SceneData);
    }
    void InitGLFW() {
        m_GlfwContext = std::unique_ptr<RTLib::Ext::GLFW::GLFWContext>(RTLib::Ext::GLFW::GLFWContext::New());
        m_GlfwContext->Initialize();
    }
    void FreeGLFW() {
        m_GlfwContext->Terminate();
        m_GlfwContext.reset();
    }
    void InitOGL4() {
        m_GlfwWindow = std::unique_ptr<RTLib::Ext::GLFW::GL::GLFWOpenGLWindow>(rtlib::test::CreateGLFWWindow(m_GlfwContext.get(), m_SceneData.config.width, m_SceneData.config.height, "title"));
    }
    void FreeOGL4() {
        m_GlfwWindow->Destroy();
        m_GlfwWindow.reset();
    }
    void InitOPX7() {
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
    void FreeOPX7() {
        m_Opx7Context->Terminate();
        m_Opx7Context.reset();
    }
    void InitPtx() {
        m_PtxStringMap["SimpleKernel.ptx"] = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH"/SimpleKernel.ptx");
    }
    void InitWorld() {
        m_ShaderTableLayout = rtlib::test::GetShaderTableLayout(m_SceneData.world, RAY_TYPE_COUNT);
        auto accelBuildOptions = OptixAccelBuildOptions();
        {
            accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accelBuildOptions.motionOptions = {};
            accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        }
        m_SceneData.InitExtData(m_Opx7Context.get());
        m_TextureMap    = m_SceneData.LoadTextureMap(m_Opx7Context.get());
        m_GeometryASMap = m_SceneData.BuildGeometryASs(m_Opx7Context.get(), accelBuildOptions);
        m_InstanceASMap = m_SceneData.BuildInstanceASs(m_Opx7Context.get(), accelBuildOptions, m_ShaderTableLayout.get(), m_GeometryASMap);
    }
    void FreeWorld() {
        {
            for (auto& [objName, objAsset] : m_SceneData.objAssetManager.GetAssets())
            {
                for (auto& [uniqueName, uniqueRes] : objAsset.meshGroup->GetUniqueResources()) {
                    reinterpret_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(uniqueRes.get())->Destroy();
                    uniqueRes->extData.reset();
                }
            }
        }
        {
            for (auto& [name, texture] : m_TextureMap) {
                texture.Destroy();
            }
            m_TextureMap.clear();
        }
        {
            for (auto& [name, geometryAS] : m_GeometryASMap) {
                auto& [buffer, handle] = geometryAS;
                buffer->Destroy();
            }
            m_GeometryASMap.clear();
        }
        {
            for (auto& [name, instanceAS] : m_InstanceASMap) {
                auto& [buffer, handle, instanceBuffer, instanceArray] = instanceAS;
                instanceBuffer->Destroy();
                buffer->Destroy();
            }
            m_InstanceASMap.clear();
        }
    }
    void InitLight() {
        m_lightBuffer = rtlib::test::UploadBuffer<MeshLight>();

        {
            for (auto& instancePath : m_ShaderTableLayout->GetInstanceNames())
            {
                auto& instanceDesc = m_ShaderTableLayout->GetDesc(instancePath);
                auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
                auto  geometryAS = instanceData->GetDwGeometryAS();
                if (geometryAS)
                {
                    for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                    {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes) {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshName);
                            auto desc = m_ShaderTableLayout->GetDesc(instancePath + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            auto meshLight = MeshLight();
                            bool hasLight = false;
                            bool useNEE = false;
                            if (mesh->GetUniqueResource()->variables.HasBool("hasLight")) {
                                hasLight = mesh->GetUniqueResource()->variables.GetBool("hasLight");
                            }
                            if (hasLight) {
                                auto meshLight = MeshLight();
                                meshLight.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                meshLight.normals = reinterpret_cast<float3*>(extSharedData->GetNormalBufferGpuAddress());
                                meshLight.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                meshLight.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                meshLight.indCount = mesh->GetUniqueResource()->triIndBuffer.size();
                                meshLight.emission = meshData.materials.front().GetFloat3As<float3>("emitCol");
                                auto emitTexStr = meshData.materials.front().GetString("emitTex");
                                if (emitTexStr == "") {
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
    void FreeLight() {
        m_lightBuffer.gpuHandle->Destroy();
        m_lightBuffer.gpuHandle.reset();
    }
    void InitPtxString() {
        m_PtxStringMap["SimpleKernel.ptx"] = rtlib::test::LoadShaderSource(RTLIB_EXT_OPX7_TEST_CUDA_PATH"/SimpleKernel.ptx");
    }
    void InitPipelines() {
        
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
            m_PipelineMap["NEE"].SetProgramGroupRG(m_Opx7Context.get(),   "SimpleKernel.NEE", "__raygen__rg");
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

                    shaderTableDesc.raygenRecordSizeInBytes     = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
                    shaderTableDesc.missRecordStrideInBytes     = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
                    shaderTableDesc.missRecordCount             = m_ShaderTableLayout->GetRecordStride();
                    shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
                    shaderTableDesc.hitgroupRecordCount         = m_ShaderTableLayout->GetRecordCount();
                    shaderTableDesc.exceptionRecordSizeInBytes  = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<unsigned int>);
                }
                m_PipelineMap["NEE"].shaderTable = std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
            }
            auto programGroupHGNames = std::vector<std::string>{
                "SimpleKernel.Radiance",
                "SimpleKernel.Occluded"
            };
            auto raygenRecord = m_PipelineMap["NEE"].programGroupRG->GetRecord<RayGenData>();
            m_PipelineMap["NEE"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
            m_PipelineMap["NEE"].SetHostMissRecordTypeData(RAY_TYPE_RADIANCE, "SimpleKernel.Radiance", MissData{});
            m_PipelineMap["NEE"].SetHostMissRecordTypeData(RAY_TYPE_OCCLUDED, "SimpleKernel.Occluded", MissData{});
            m_PipelineMap["NEE"].SetHostExceptionRecordTypeData(unsigned int());
            for (auto& instanceName : m_ShaderTableLayout->GetInstanceNames())
            {
                auto& instanceDesc = m_ShaderTableLayout->GetDesc(instanceName);
                auto* instanceData = reinterpret_cast<const RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance*>(instanceDesc.pData);
                auto  geometryAS = instanceData->GetDwGeometryAS();
                if (geometryAS)
                {
                    for (auto& geometryName : m_SceneData.world.geometryASs[geometryAS->GetName()].geometries)
                    {
                        auto& geometry = m_SceneData.world.geometryObjModels[geometryName];
                        auto objAsset = m_SceneData.objAssetManager.GetAsset(geometry.base);
                        for (auto& [meshName, meshData] : geometry.meshes) {
                            auto mesh = objAsset.meshGroup->LoadMesh(meshName);
                            auto desc = m_ShaderTableLayout->GetDesc(instanceName + "/" + meshName);
                            auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                            auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                            for (auto i = 0; i < desc.recordCount; ++i)
                            {
                                auto hitgroupData = HitgroupData();
                                if ((i % desc.recordStride) == RAY_TYPE_RADIANCE) {
                                    auto material = meshData.materials[i / desc.recordStride];
                                    hitgroupData.vertices = reinterpret_cast<float3*>(extSharedData->GetVertexBufferGpuAddress());
                                    hitgroupData.indices = reinterpret_cast<uint3*>(extUniqueData->GetTriIdxBufferGpuAddress());
                                    hitgroupData.texCrds = reinterpret_cast<float2*>(extSharedData->GetTexCrdBufferGpuAddress());
                                    hitgroupData.diffuse = material.GetFloat3As<float3>("diffCol");
                                    hitgroupData.emission = material.GetFloat3As<float3>("emitCol");
                                    hitgroupData.specular = material.GetFloat3As<float3>("specCol");
                                    auto diffTexStr = material.GetString("diffTex");
                                    if (diffTexStr == "") {
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

            m_PipelineMap["NEE"].pipeline->SetStackSize(0, 0, continuationStackSizes, m_ShaderTableLayout->GetMaxTraversableDepth());
            auto params = Params();
            {

            }
            m_PipelineMap["NEE"].InitParams(m_Opx7Context.get(), sizeof(Params), &params);
        }
    }
    int OnlineSample()
    {
        this->LoadScene();
        using namespace RTLib::Ext::CUDA;
        using namespace RTLib::Ext::OPX7;
        using namespace RTLib::Ext::GL;
        using namespace RTLib::Ext::CUGL;
        using namespace RTLib::Ext::GLFW;
        int  width = m_SceneData.config.width;
        int  height = m_SceneData.config.height;
        int  maxDepth = m_SceneData.config.maxDepth;
        int  samplesForLaunch = m_SceneData.config.samples;
        int  maxSamples = m_SceneData.config.maxSamples;
        int  samplesForAccum = 0;
        auto& cameraController = m_SceneData.cameraController;
        auto& objAssetLoader   = m_SceneData.objAssetManager;
        auto& worldData        = m_SceneData.world;
        try
        {
            this->InitGLFW();
            this->InitOGL4();
            this->InitOPX7();
            this->InitWorld();
            this->InitLight();
            this->InitPtxString();
            this->InitPipelines();

            auto  seedBufferCUDA = std::unique_ptr<CUDABuffer>();
            {
                auto mt19937 = std::mt19937();
                auto seedData = std::vector<unsigned int>(width * height * sizeof(unsigned int));
                std::generate(std::begin(seedData), std::end(seedData), mt19937);
                seedBufferCUDA = std::unique_ptr<CUDABuffer>(m_Opx7Context->CreateBuffer(
                    { CUDAMemoryFlags::eDefault,seedData.size() * sizeof(seedData[0]),seedData.data() }
                ));
            }

            auto ogl4Context     = m_GlfwWindow->GetOpenGLContext();
            auto accumBufferCUDA = std::unique_ptr<CUDABuffer>(m_Opx7Context->CreateBuffer({ CUDAMemoryFlags::eDefault,width * height * sizeof(float) * 3,nullptr }));
            auto frameBufferGL   = std::unique_ptr<GLBuffer  >(ogl4Context->CreateBuffer(GLBufferCreateDesc{ sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr }));
            auto frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(m_Opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
            auto frameTextureGL  = rtlib::test::CreateFrameTextureGL(ogl4Context, width, height);
            auto rectRendererGL  = std::unique_ptr<GLRectRenderer>(ogl4Context->CreateRectRenderer({ 1, false, true }));

            auto stream = std::unique_ptr<CUDAStream>(m_Opx7Context->CreateStream());

            auto eventState = EventState();
            auto windowState = WindowState();
            {
                m_GlfwWindow->Show();
                m_GlfwWindow->SetResizable(true);
                m_GlfwWindow->SetUserPointer(&windowState);
                //window->SetCursorPosCallback(cursorPosCallback);
            }
            while (!m_GlfwWindow->ShouldClose())
            {
                if (samplesForAccum >= maxSamples) {
                    break;
                }
                {
                    if (eventState.isResized)
                    {
                        frameBufferCUGL->Destroy();
                        frameBufferGL->Destroy();
                        frameTextureGL->Destroy();
                        frameBufferGL = std::unique_ptr<GLBuffer>(ogl4Context->CreateBuffer(GLBufferCreateDesc{ sizeof(uchar4) * width * height, GLBufferUsageImageCopySrc, GLMemoryPropertyDefault, nullptr }));
                        frameBufferCUGL = std::unique_ptr<CUGLBuffer>(CUGLBuffer::New(m_Opx7Context.get(), frameBufferGL.get(), CUGLGraphicsRegisterFlagsWriteDiscard));
                        frameTextureGL = rtlib::test::CreateFrameTextureGL(ogl4Context, width, height);
                    }
                    if (eventState.isUpdated) {
                        auto zeroClearData = std::vector<float>(width * height * 3, 0.0f);
                        m_Opx7Context->CopyMemoryToBuffer(accumBufferCUDA.get(), { {zeroClearData.data(), 0, sizeof(zeroClearData[0]) * zeroClearData.size()} });
                        samplesForAccum = 0;
                    }
                    if (eventState.isMovedCamera)
                    {
                        m_PipelineMap["NEE"].SetHostRayGenRecordTypeData(rtlib::test::GetRaygenData(m_SceneData.GetCamera()));
                        m_PipelineMap["NEE"].shaderTable->UploadRaygenRecord();
                    }
                }
                {

                    auto frameBufferCUDA = frameBufferCUGL->Map(stream.get());
                    /*RayTrace*/
                    {
                        auto params = Params();
                        {
                            params.accumBuffer = reinterpret_cast<float3*>(CUDANatives::GetCUdeviceptr(accumBufferCUDA.get()));
                            params.seedBuffer  = reinterpret_cast<unsigned int*>(CUDANatives::GetCUdeviceptr(seedBufferCUDA.get()));
                            params.frameBuffer = reinterpret_cast<uchar4*>(CUDANatives::GetCUdeviceptr(frameBufferCUDA));
                            params.width       = width;
                            params.height      = height;
                            params.maxDepth    = maxDepth;
                            params.flags       = PARAM_FLAG_NEE;
                            params.samplesForAccum  = samplesForAccum;
                            params.samplesForLaunch = samplesForLaunch;
                            params.gasHandle = m_InstanceASMap["Root"].handle;
                            params.lights.count = m_lightBuffer.cpuHandle.size();
                            params.lights.data = reinterpret_cast<MeshLight*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_lightBuffer.gpuHandle.get()));
                        }
                        stream->CopyMemoryToBuffer(m_PipelineMap["NEE"].paramsBuffer.get(), { {&params, 0, sizeof(params)} });
                        m_PipelineMap["NEE"].Launch(stream.get(), width, height);
                        samplesForAccum += samplesForLaunch;
                    }
                    frameBufferCUGL->Unmap(stream.get());
                    stream->Synchronize();
                }
                /*DrawRect*/
                rtlib::test::RenderFrameGL(ogl4Context, rectRendererGL.get(), frameBufferGL.get(), frameTextureGL.get());

                m_GlfwContext->Update();

                eventState = EventState();
                {
                    int tWidth, tHeight;
                    glfwGetWindowSize(m_GlfwWindow->GetGLFWwindow(), &tWidth, &tHeight);
                    if (width != tWidth || height != tHeight)
                    {
                        std::cout << width << "->" << tWidth << "\n";
                        std::cout << height << "->" << tHeight << "\n";
                        width = tWidth;
                        height = tHeight;
                        eventState.isResized = true;
                        eventState.isUpdated = true;
                    }
                    else
                    {
                        eventState.isResized = false;
                    }
                    float prevTime = glfwGetTime();
                    {
                        windowState.delTime = windowState.curTime - prevTime;
                        windowState.curTime = prevTime;
                    }
                    eventState.isMovedCamera = rtlib::test::UpdateCameraMovement(
                        cameraController,
                        m_GlfwWindow.get(),
                        windowState.delTime,
                        windowState.delCurPos.x,
                        windowState.delCurPos.y
                    );
                    if (eventState.isMovedCamera) {
                        eventState.isUpdated = true;
                    }
                }

                m_GlfwWindow->SwapBuffers();
            }

            this->SaveScene();

            {
                stream->Synchronize();
                
                accumBufferCUDA->Destroy();
                frameBufferCUGL->Destroy();
                frameBufferGL->Destroy();
                stream->Destroy();
                this->FreeLight();
                this->FreeWorld();
                this->FreeOGL4();
                this->FreeGLFW();
                this->FreeOPX7();
            }
        }
        catch (std::runtime_error& err)
        {
            std::cerr << err.what() << std::endl;
        }
        return 0;
    }
private:
    std::string                                                                    m_ScenePath;
    rtlib::test::SceneData                                                         m_SceneData;
    std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTableLayout>                       m_ShaderTableLayout;
    std::unordered_map<std::string, std::vector<char>>                             m_PtxStringMap;
    std::unordered_map<std::string, std::unique_ptr<RTLib::Ext::OPX7::OPX7Module>> m_ModuleMap;
    std::unique_ptr<RTLib::Ext::OPX7::OPX7Context>                                 m_Opx7Context;
    std::unique_ptr<RTLib::Ext::GLFW::GLFWContext>                                 m_GlfwContext;
    std::unique_ptr<RTLib::Ext::GLFW::GL::GLFWOpenGLWindow>                        m_GlfwWindow;
    std::unordered_map<std::string,rtlib::test::TextureData>                       m_TextureMap;
    std::unordered_map<std::string,rtlib::test::GeometryAccelerationStructureData> m_GeometryASMap;
    std::unordered_map<std::string,rtlib::test::InstanceAccelerationStructureData> m_InstanceASMap;
    std::unordered_map<std::string, rtlib::test::PipelineData>                     m_PipelineMap;
    rtlib::test::UploadBuffer<MeshLight>                                           m_lightBuffer;
};
int main()
{
    return RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene2.json").OnlineSample();
}

