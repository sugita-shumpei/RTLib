#ifndef RTLIB_EXT_OPX_7_TEST_APPLICATION_H
#define RTLIB_EXT_OPX_7_TEST_APPLICATION_H
#include <RTLibExtOPX7Test.h>
class RTLibExtOPX7TestApplication
{
public:
    RTLibExtOPX7TestApplication(const std::string& scenePath, std::string defPipelineName, bool enableVis = true, bool enableGrid = false,bool enableTree = false) noexcept
    {
        m_CurPipelineName = defPipelineName;
        m_ScenePath       = scenePath;
        m_EnableVis       = enableVis;
        m_EnableGrid      = enableGrid;
        m_EnableTree      = enableTree;
    }

    void Initialize()
    {
        this->InitOPX7();
        this->LoadScene();
        this->InitWorld();
        this->InitLight();
        this->InitGrids();
        if (m_EnableTree) {
            this->InitSdTree();
        }
        this->InitPtxString();
        this->InitDefPipeline();
        this->InitNeePipeline();
        this->InitDbgPipeline();
        this->InitRisPipeline();
        this->InitSdTreeDefPipeline();
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
        m_EventState      = rtlib::test::EventState();
        m_WindowState     = rtlib::test::WindowState();
        m_EventState.isClearFrame = true;
        m_SamplesForAccum = 0;
        m_TimesForAccum = 0;
        this->UpdateTimeStamp();
        while (!this->FinishTrace())
        {
            this->UpdateTrace();
            /*RayTrace*/
            this->TraceFrame(m_Stream.get());
            /*DrawRect*/
            this->UpdateState();
        }

        m_Stream->Synchronize();
        m_Stream->Destroy();
        m_Stream.reset();

        {
            m_HashBufferCUDA.Download(m_Opx7Context.get());
            float v = 0.0f;
            for (auto& gridVal : m_HashBufferCUDA.dataCpuHandle) {
                if (gridVal.w != 0.0f) {
                    v += 1.0f;
                }
            }
            v /= static_cast<float>(m_HashBufferCUDA.dataCpuHandle.size());
            std::cout << "Capacity: " << v * 100.0f << "%" << std::endl;
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
        this->FreePipelines();
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

    auto GetWidth ()const noexcept -> unsigned int { return m_SceneData.config.width ; }
    auto GetHeight()const noexcept -> unsigned int { return m_SceneData.config.height; }

    void SetWidth (unsigned int width  ) noexcept  { m_SceneData.config.width = width ; }
    void SetHeight(unsigned int height ) noexcept  { m_SceneData.config.height= height; }

    auto GetPipelineName()const noexcept -> std::string { return m_CurPipelineName; }
    void SetPipelineName(const std::string pipelineName)noexcept { m_CurPipelineName = pipelineName; }

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
    }
private:
    static void CursorPosCallback(RTLib::Core::Window* window, double x, double y);

    void LoadScene();
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

    void InitDefPipeline();
    void InitNeePipeline();
    void InitDbgPipeline();
    void InitRisPipeline();
    void InitSdTreeDefPipeline();

    void FreePipelines();

    void InitFrameResourceCUDA();
    void FreeFrameResourceCUDA();

    void InitFrameResourceCUGL();
    void FreeFrameResourceCUGL();

    void InitFrameResourceOGL4();
    void FreeFrameResourceOGL4();

    void InitRectRendererGL();
    void FreeRectRendererGL();

    void InitWindowCallback();

    void TracePipeline(RTLib::Ext::CUDA::CUDAStream* stream, RTLib::Ext::CUDA::CUDABuffer* frameBuffer);
    
    bool FinishTrace();
    void UpdateTrace();
    void UpdateState();

    void UpdateTimeStamp();
    void TraceFrame(RTLib::Ext::CUDA::CUDAStream* stream);

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
    std::unordered_map<std::string, rtlib::test::PipelineData> m_PipelineMap;

    std::unique_ptr<RTLib::Ext::CUDA::CUDAStream> m_Stream;
    rtlib::test::UploadBuffer<MeshLight>          m_lightBuffer;
    std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_AccumBufferCUDA;
    std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_FrameBufferCUDA;
    std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_SeedBufferCUDA;
    rtlib::test::HashGrid3Buffer<float4>          m_HashBufferCUDA;
    std::unique_ptr<rtlib::test::RTSTreeWrapper>  m_SdTree;

    std::unique_ptr<RTLib::Ext::GLFW::GL::GLFWOpenGLWindow> m_GlfwWindow;
    std::unique_ptr<RTLib::Ext::GL::GLRectRenderer>         m_RectRendererGL;
    std::unique_ptr<RTLib::Ext::GL::GLBuffer>               m_FrameBufferGL;
    std::unique_ptr<RTLib::Ext::GL::GLTexture>              m_FrameTextureGL;
    std::unique_ptr<RTLib::Ext::CUGL::CUGLBuffer>           m_FrameBufferCUGL;


    std::string m_CurPipelineName = "DEF";
    std::string m_PrvPipelineName = "DEF";

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