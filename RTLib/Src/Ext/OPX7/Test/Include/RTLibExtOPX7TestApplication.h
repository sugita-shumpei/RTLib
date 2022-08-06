#ifndef RTLIB_EXT_OPX_7_TEST_APPLICATION_H
#define RTLIB_EXT_OPX_7_TEST_APPLICATION_H
#include <RTLibExtOPX7Test.h>
class RTLibExtOPX7TestApplication
{
public:
    RTLibExtOPX7TestApplication(const std::string& scenePath, std::string defTracerName, bool enableVis = true, bool enableGrid = false,bool enableTree = false) noexcept
    {
        m_CurTracerName = defTracerName;
        m_ScenePath       = scenePath;
        m_EnableVis       = enableVis;
        m_EnableGrid      = enableGrid;
        m_EnableTree      = enableTree;
    }

    void Initialize(int argc = 0, const char** argv = nullptr)
    {
        this->InitOPX7();
        this->LoadScene(argc,argv);
        this->InitWorld();
        this->InitLight();
        this->InitGrids();
        if (m_EnableTree) {
            this->InitSdTree();
        }
        this->InitPtxString();
        this->InitDefTracer();
        this->InitNeeTracer();
        this->InitDbgTracer();
        this->InitRisTracer();
        this->InitSdTreeDefTracer();
        this->InitSdTreeNeeTracer();
        this->InitSdTreeRisTracer();
        this->InitHashTreeDefTracer();
        this->InitHashTreeNeeTracer();
        this->InitHashTreeRisTracer();
        this->InitFrameResourceCUDA();
        if (m_EnableVis)
        {
            this->InitGLFW();
            this->InitOGL4();
            this->InitFrameResourceOGL4();
            this->InitFrameResourceCUGL();
            this->InitRectRendererGL();
            this->InitWindowCallback();
        }
    }

    void MainLoop()
    {
        m_Stream          = std::unique_ptr<RTLib::Ext::CUDA::CUDAStream>(m_Opx7Context->CreateStream());
        if (m_EnableVis)
        {
            m_KeyBoardManager = std::make_unique<rtlib::test::KeyBoardStateManager>(m_GlfwWindow.get());

            for (int i = 0; i < DEBUG_FRAME_TYPE_COUNT; ++i) {
                m_KeyBoardManager->UpdateState(GLFW_KEY_1    + i);
                m_KeyBoardManager->UpdateState(GLFW_KEY_KP_1 + i);
            }

            m_KeyBoardManager->UpdateState(GLFW_KEY_F1);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F2);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F3);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F4);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F5);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F6);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F7);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F8);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F9);
            m_KeyBoardManager->UpdateState(GLFW_KEY_F10);


            m_KeyBoardManager->UpdateState(GLFW_KEY_W);
            m_KeyBoardManager->UpdateState(GLFW_KEY_A);
            m_KeyBoardManager->UpdateState(GLFW_KEY_S);
            m_KeyBoardManager->UpdateState(GLFW_KEY_D);

            m_KeyBoardManager->UpdateState(GLFW_KEY_UP);
            m_KeyBoardManager->UpdateState(GLFW_KEY_DOWN);
            m_KeyBoardManager->UpdateState(GLFW_KEY_LEFT);
            m_KeyBoardManager->UpdateState(GLFW_KEY_RIGHT);

            m_MouseButtonManager = std::make_unique<rtlib::test::MouseButtonStateManager>(m_GlfwWindow.get());
            m_MouseButtonManager->UpdateState(GLFW_MOUSE_BUTTON_LEFT);

            m_GlfwWindow->Show();
        }
        {
            m_EventState              = rtlib::test::EventState();
            m_WindowState             = rtlib::test::WindowState();
            m_EventState.isClearFrame = true;
            m_SamplesForAccum         = 0;
            m_TimesForAccum           = 0;
            this->SetupPipeline();
        }
        {
            m_EventState = rtlib::test::EventState();
            m_WindowState = rtlib::test::WindowState();
            m_EventState.isClearFrame = true;
            m_SamplesForAccum = 0;
            m_TimesForAccum = 0;

            if ((m_CurTracerName == "PGDEF") ||
                (m_CurTracerName == "PGNEE") ||
                (m_CurTracerName == "PGRIS")) {
                if (m_EnableTree) {
                    m_SdTreeController->Start();
                }

            }

            if ((m_CurTracerName == "HTDEF") ||
                (m_CurTracerName == "HTNEE") ||
                (m_CurTracerName == "HTRIS")) {
                if (m_EnableGrid) {
                    m_MortonQuadTreeController->Start();
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        this->UpdateTimeStamp();
        while (!this->FinishTrace())
        {
            this->UpdateTrace();
            /*RayTrace*/
            this->TraceFrame(m_Stream.get());
            /*DrawRect*/
            this->UpdateState();
        }
        this->SaveResultImage(m_Stream.get());
        m_Stream->Synchronize();
        m_Stream->Destroy();
        m_Stream.reset();

        {
            m_HashBufferCUDA.Download(m_Opx7Context.get());
            float v = 0.0f;
            for (auto& gridVal : m_HashBufferCUDA.checkSumCpuHandle) {
                if (gridVal != 0.0f) {
                    v += 1.0f;
                }
            }
            v /= static_cast<float>(m_HashBufferCUDA.checkSumCpuHandle.size());
            std::cout << "Capacity: " << v * 100.0f << "%" << std::endl;
        }
        if (m_EnableTree) {
            std::cout << "SdTreeMemory(MB): " << (m_SdTree->GetMemoryFootPrint()     ) / static_cast<float>(1000 * 1000) << std::endl;
            std::cout << " STreeMemory(MB): "  << (m_SdTree->GetSTreeMemoryFootPrint()) / static_cast<float>(1000 * 1000) << std::endl;
            std::cout << " DTreeMemory(MB): "  << (m_SdTree->GetDTreeMemoryFootPrint()) / static_cast<float>(1000 * 1000) << std::endl;
        }
        if (m_EnableGrid) {
            auto hashGridMemoryFootPrint    = m_HashBufferCUDA.GetMemoryFootPrint();
            auto dTreeMemoryFootPrint = m_MortonQuadTree->GetMemoryFootPrint();
            std::cout << "HTreeMemory(MB): " << (dTreeMemoryFootPrint + hashGridMemoryFootPrint) / static_cast<float>(1000 * 1000) << std::endl;
            std::cout << "HGridMemory(MB): " << hashGridMemoryFootPrint / static_cast<float>(1000 * 1000) << std::endl;
            std::cout << "DTreeMemory(MB): " << dTreeMemoryFootPrint / static_cast<float>(1000 * 1000) << std::endl;
        }
    }

    void Terminate()
    {
        this->SaveScene();
        this->FreeRectRendererGL();
        this->FreeFrameResourceCUGL();
        this->FreeFrameResourceOGL4();
        this->FreeOGL4();
        this->FreeGLFW();
        this->FreeFrameResourceCUDA();
        this->FreeTracers();
        if (m_EnableTree) {
            this->FreeSdTree();
        }
        this->FreeGrids();
        this->FreeLight();
        this->FreeWorld();
        this->FreeOPX7();
    }

    int Run()
    {
        try
        {
            this->Initialize();
            this->MainLoop();
            this->Terminate();
        }
        catch (std::runtime_error& err)
        {
            std::cerr << err.what() << std::endl;
        }
        return 0;
    }

    void ResetGrids() {
        if (!m_EnableGrid) {
            return;
        }
        FreeGrids();
        InitGrids();
    }

    void ResetSdTree() {
        if (!m_EnableTree) {
            return;
        }
        FreeSdTree();
        InitSdTree();
    }

    auto GetWidth ()const noexcept -> unsigned int { return m_SceneData.config.width ; }
    auto GetHeight()const noexcept -> unsigned int { return m_SceneData.config.height; }

    void SetWidth (unsigned int width  ) noexcept  { m_SceneData.config.width = width ; }
    void SetHeight(unsigned int height ) noexcept  { m_SceneData.config.height= height; }

    auto GetTracerName()const noexcept -> std::string { return m_CurTracerName; }
    void SetTracerName(const std::string tracerName)noexcept { m_CurTracerName = tracerName; }

    auto GetSamplesPerSave()const noexcept -> unsigned int {
        return m_SceneData.config.samplesPerSave;
    }
    void SetSamplesPerSave(unsigned int samplesPerSave)noexcept {
        m_SceneData.config.samplesPerSave = samplesPerSave;
        
    }

    auto GetMaxSamples()const noexcept -> unsigned int {
        return m_SceneData.config.maxSamples;
    }
    void SetMaxSamples(unsigned int maxSamples)
        noexcept {
        m_SceneData.config.maxSamples = maxSamples;
        if (m_EnableGrid) {
            m_MortonQuadTreeController->SetSampleForBudget(maxSamples);
        }
    }

    auto GetMaxTimes()const noexcept -> float {
        return m_SceneData.config.maxTimes;
    }
    void SetMaxTimes(float maxTimes)
        noexcept {
        m_SceneData.config.maxTimes = maxTimes;
    }

    auto GetMaxDepth()const noexcept -> unsigned int { return m_SceneData.config.maxDepth; }
    void SetMaxDepth(unsigned int maxDepth)noexcept { m_SceneData.config.maxDepth = maxDepth; }

    auto GetTraceConfig()const noexcept -> const rtlib::test::TraceConfigData& { return m_SceneData.config; }
    auto GetTraceConfig()      noexcept ->       rtlib::test::TraceConfigData& { return m_SceneData.config; }
private:
    static void CursorPosCallback(RTLib::Core::Window* window, double x, double y);

    void LoadScene(int argc = 0, const char** argv = nullptr);
    void SaveScene();

    void InitGLFW();
    void FreeGLFW();

    void InitOGL4();
    void FreeOGL4();

    void InitOPX7();
    void FreeOPX7();

    void InitWorld();
    void FreeWorld();

    void InitLight();
    void FreeLight();

    void InitGrids();
    void FreeGrids();

    void InitSdTree();
    void FreeSdTree();

    void InitPtxString();

    void InitDefTracer();
    void InitNeeTracer();
    void InitRisTracer();
    void InitDbgTracer();

    void InitSdTreeDefTracer();
    void InitSdTreeNeeTracer();
    void InitSdTreeRisTracer();

    void InitHashTreeDefTracer();
    void InitHashTreeNeeTracer();
    void InitHashTreeRisTracer();

    void FreeTracers();

    void InitFrameResourceCUDA();
    void FreeFrameResourceCUDA();

    void InitFrameResourceCUGL();
    void FreeFrameResourceCUGL();

    void InitFrameResourceOGL4();
    void FreeFrameResourceOGL4();

    void InitRectRendererGL();
    void FreeRectRendererGL();

    void InitWindowCallback();

    bool TracePipeline(RTLib::Ext::CUDA::CUDAStream* stream, RTLib::Ext::CUDA::CUDABuffer* frameBuffer);
    void SetupPipeline();
    
    bool FinishTrace();
    void UpdateTrace();
    void UpdateState();

    void UpdateTimeStamp();
    void TraceFrame(RTLib::Ext::CUDA::CUDAStream* stream);
    void SaveResultImage(RTLib::Ext::CUDA::CUDAStream* stream);
private:
    auto NewShaderTable()->std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>
    {
        auto shaderTableDesc = RTLib::Ext::OPX7::OPX7ShaderTableCreateDesc();
        shaderTableDesc.raygenRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<RayGenData>);
        shaderTableDesc.missRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<MissData>);
        shaderTableDesc.missRecordCount = m_ShaderTableLayout->GetRecordStride();
        shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<HitgroupData>);
        shaderTableDesc.hitgroupRecordCount = m_ShaderTableLayout->GetRecordCount();
        shaderTableDesc.exceptionRecordSizeInBytes = sizeof(RTLib::Ext::OPX7::OPX7ShaderRecord<unsigned int>);
        return std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>(m_Opx7Context->CreateOPXShaderTable(shaderTableDesc));
    }
    auto GetParams(RTLib::Ext::CUDA::CUDABuffer* frameBuffer) noexcept -> Params
    {
        auto params = Params();
        params.accumBuffer = reinterpret_cast<float3*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_AccumBufferCUDA.get()));
        params.seedBuffer  = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<unsigned int>(m_SeedBufferCUDA.get());
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
        params.numCandidates = GetTraceConfig().custom.GetUInt32Or("Ris.NumCandidates",32);
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
        if (m_CurTracerName == "DEF")
        {
            params.flags = PARAM_FLAG_NONE;

            if (m_EnableGrid) {
                params.flags |= PARAM_FLAG_USE_GRID;
                params.mortonTree = m_MortonQuadTree->GetGpuHandle();
            }
        }
        if (m_CurTracerName == "NEE")
        {
            params.flags      = PARAM_FLAG_NEE;

            if (m_EnableGrid) {

                params.flags |= PARAM_FLAG_USE_GRID;
                params.mortonTree = m_MortonQuadTree->GetGpuHandle();
            }
        }
        if (m_CurTracerName == "RIS") {
            params.flags      = PARAM_FLAG_NEE | PARAM_FLAG_RIS;

            if (m_EnableGrid) {
                params.flags |= PARAM_FLAG_USE_GRID;
                params.mortonTree = m_MortonQuadTree->GetGpuHandle();
            }
        }
        if ((m_CurTracerName=="PGDEF")||((m_CurTracerName=="PGNEE"))||(m_CurTracerName=="PGRIS"))
        {
            params.flags      = PARAM_FLAG_NONE;
            if (m_EnableTree) {
                params.tree   = m_SdTreeController->GetGpuSTree();
                params.flags |= PARAM_FLAG_USE_TREE;
            }
            
            if (m_CurTracerName == "PGNEE")
            {
                params.flags |= PARAM_FLAG_NEE;
            }
            if (m_CurTracerName == "PGRIS")
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
        if ((m_CurTracerName=="HTDEF")||((m_CurTracerName=="HTNEE"))||(m_CurTracerName=="HTRIS"))
        {
            if (m_EnableGrid) {
                params.mortonTree = m_MortonQuadTreeController->GetGpuHandle();
                params.grid       = m_HashBufferCUDA.GetHandle();
                params.flags     |= PARAM_FLAG_USE_GRID;
                params.mortonTree.level = RTLib::Ext::CUDA::Math::min(params.mortonTree.level, rtlib::test::MortonQTreeWrapper::kMaxTreeLevel);
            }

            if (m_CurTracerName == "HTNEE")
            {
                params.flags |= PARAM_FLAG_NEE;
            }
            if (m_CurTracerName == "HTRIS")
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
private:
    std::string m_ScenePath;
    rtlib::test::SceneData m_SceneData;
    std::unique_ptr<rtlib::test::KeyBoardStateManager> m_KeyBoardManager;
    std::unique_ptr<rtlib::test::MouseButtonStateManager> m_MouseButtonManager;
    std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTableLayout> m_ShaderTableLayout;
    std::unordered_map<std::string, std::vector<char>> m_PtxStringMap;
    std::unordered_map<std::string, std::unique_ptr<RTLib::Ext::OPX7::OPX7Module>> m_ModuleMap;
    std::unique_ptr<RTLib::Ext::OPX7::OPX7Context> m_Opx7Context;
    std::unique_ptr<RTLib::Ext::GLFW::GLFWContext> m_GlfwContext;
    std::unordered_map<std::string, rtlib::test::TextureData> m_TextureMap;
    std::unordered_map<std::string, rtlib::test::GeometryAccelerationStructureData> m_GeometryASMap;
    std::unordered_map<std::string, rtlib::test::InstanceAccelerationStructureData> m_InstanceASMap;
    std::unordered_map<std::string, rtlib::test::TracerData> m_TracerMap;

    std::unique_ptr<RTLib::Ext::CUDA::CUDAStream>   m_Stream;
    rtlib::test::UploadBuffer<MeshLight>            m_lightBuffer;
    std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>   m_AccumBufferCUDA;
    std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>   m_FrameBufferCUDA;
    std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>   m_SeedBufferCUDA;
    rtlib::test::DoubleBufferedHashGrid3Buffer      m_HashBufferCUDA;
    std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>   m_DiffuseBufferCUDA;
    std::unique_ptr<rtlib::test::RTSTreeWrapper>    m_SdTree;
    std::unique_ptr<rtlib::test::RTSTreeController> m_SdTreeController;
    std::unique_ptr<rtlib::test::RTMortonQuadTreeWrapper> m_MortonQuadTree;
    std::unique_ptr<rtlib::test::RTMortonQuadTreeController> m_MortonQuadTreeController;

    std::unique_ptr<RTLib::Ext::GLFW::GL::GLFWOpenGLWindow> m_GlfwWindow;
    std::unique_ptr<RTLib::Ext::GL::GLRectRenderer>         m_RectRendererGL;
    std::unique_ptr<RTLib::Ext::GL::GLBuffer>               m_FrameBufferGL;
    std::unique_ptr<RTLib::Ext::GL::GLTexture>              m_FrameTextureGL;
    std::unique_ptr<RTLib::Ext::CUGL::CUGLBuffer>           m_FrameBufferCUGL;

    std::string m_CurTracerName = "DEF";
    std::string m_PrvTracerName = "DEF";
    std::string m_PipelineName  = "Trace";

    unsigned int m_DebugFrameType = DEBUG_FRAME_TYPE_NORMAL;

    rtlib::test::EventState m_EventState = rtlib::test::EventState();
    rtlib::test::WindowState m_WindowState = rtlib::test::WindowState();

    std::array<float, 3> m_WorldAabbMin = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    std::array<float, 3> m_WorldAabbMax = { FLT_MAX, FLT_MAX, FLT_MAX };

    int   m_SamplesForAccum = 0;

    unsigned long long m_TimesForAccum = 0;
    unsigned long long m_TimesForFrame = 0;

    std::string m_TimeStampString = "";

    bool  m_EnableVis  = true;
    bool  m_EnableGrid = false;
    bool  m_EnableTree = false;
};
#endif