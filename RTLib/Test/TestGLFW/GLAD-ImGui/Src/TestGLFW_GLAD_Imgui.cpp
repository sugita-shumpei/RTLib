#include <TestGLFW_GLAD_Imgui.h>
#include <vector>
#include <utility>
#include <string>
struct RTLib::Test::TestGLFWGLADImGuiAppExtendedData::Impl{
    bool  m_IsImguiInit = false;
    float m_X           = 0.0f;
    float m_Y           = 0.0f;
};
RTLib::Test::TestGLFWGLADImGuiAppExtendedData::TestGLFWGLADImGuiAppExtendedData(TestLib::TestApplication* app)noexcept
:Test::TestGLFWAppExtendedData(static_cast<Test::TestGLFWApplication*>(app)){
    m_Impl = std::unique_ptr<RTLib::Test::TestGLFWGLADImGuiAppExtendedData::Impl>(new RTLib::Test::TestGLFWGLADImGuiAppExtendedData::Impl());
}

RTLib::Test::TestGLFWGLADImGuiAppExtendedData::~TestGLFWGLADImGuiAppExtendedData()noexcept{
    m_Impl.reset();
}

void RTLib::Test::TestGLFWGLADImGuiAppExtendedData::InitImGui(){
    if (m_Impl->m_IsImguiInit){
        return;
    }
    
    if (!GetParent()){
        return;
    }
//    auto app = static_cast<const RTLib::Test::TestGLFWApplication*>(GetParent());
    int glVersionMajor = 0;
    int glVersionMinor = 0;
    glGetIntegerv(GL_MAJOR_VERSION,&glVersionMajor);
    glGetIntegerv(GL_MINOR_VERSION,&glVersionMinor);
    
    std::string glsl_version = std::string("#version ")+std::to_string(100*glVersionMajor+10*glVersionMinor);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOther(GetWindow(), true);
    ImGui_ImplOpenGL3_Init(glsl_version.data());
    m_Impl->m_IsImguiInit = true;
}
void RTLib::Test::TestGLFWGLADImGuiAppExtendedData::FreeImGui()noexcept{
    if (!m_Impl->m_IsImguiInit){
        return;
    }
    if (!GetParent()){
        return;
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    m_Impl->m_IsImguiInit = false;
}
void RTLib::Test::TestGLFWGLADImGuiAppExtendedData::DrawImGui(){
    if (!m_Impl->m_IsImguiInit){
        return;
    }
    if (!GetParent()){
        return;
    }
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Hello, world!");

    ImGui::Text("This is some useful text.");
    ImGui::DragFloat("x", &m_Impl->m_X);
    ImGui::DragFloat("y", &m_Impl->m_Y);

    ImGui::End();

     // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void RTLib::Test::TestGLFWGLADImGuiAppInitDelegate::Init()
{
    InitGLFW();
    InitGLWindow();
    InitGLAD();
    InitImGui();
    ShowWindow();
}
void RTLib::Test::TestGLFWGLADImGuiAppInitDelegate::InitGLWindow()
{
    auto windowHints = std::unordered_map<int, int>();
    windowHints[GLFW_CLIENT_API]            = GLFW_OPENGL_API;
    windowHints[GLFW_OPENGL_PROFILE]        = GLFW_OPENGL_CORE_PROFILE;
    windowHints[GLFW_OPENGL_FORWARD_COMPAT] = GLFW_TRUE;
    windowHints[GLFW_VISIBLE]               = GLFW_FALSE;
    std::vector<std::pair<int, int>> glVersions = {
        /*OpenGL 4.x*/{4,6},{4,5},{4,4},{4,3},{4,2},{4,1},{4,0},
        /*OpenGL 3.x*/{3,3},{3,2},{3,1},{3,0},
        /*OpenGL 2.x*/{2,1},{2,0}
    };

    for (auto& [version_major, version_minor] : glVersions) {
        bool isSuccess = true;
        try {
            windowHints[GLFW_CONTEXT_VERSION_MAJOR] = version_major;
            windowHints[GLFW_CONTEXT_VERSION_MINOR] = version_minor;
            InitWindow(windowHints);
        }
        catch (std::exception& err) {
            isSuccess = false;
        }
        if (isSuccess) {
            break;
        }
    }
}
void RTLib::Test::TestGLFWGLADImGuiAppInitDelegate::InitGLAD()
{
    MakeContext();
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed To Initialize GLAD!");
    }
}
void RTLib::Test::TestGLFWGLADImGuiAppInitDelegate::InitImGui()
{
    if (!GetParent()){
        return;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtendedData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWGLADImGuiAppExtendedData*>(app->GetExtendedData());
    appExtData->InitImGui();
}
void RTLib::Test::TestGLFWGLADImGuiAppMainDelegate::Main() {
    while (!ShouldClose()) {
        PollEvents ();
        RenderFrame();
        RenderImGui();
        SwapBuffers();
    }
}
void RTLib::Test::TestGLFWGLADImGuiAppMainDelegate::RenderFrame() {
    int display_w, display_h;
    glfwGetFramebufferSize(GetWindow(), &display_w, &display_h);
    glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, display_w, display_h);

}
void RTLib::Test::TestGLFWGLADImGuiAppMainDelegate::RenderImGui() {
    
    if (!GetParent()){
        return;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtendedData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWGLADImGuiAppExtendedData*>(app->GetExtendedData());
    appExtData->DrawImGui();
}

void RTLib::Test::TestGLFWGLADImGuiAppFreeDelegate::Free()noexcept{
    if (!GetParent()){
        return;
    }
    FreeImGui();
    FreeWindow();
    FreeGLFW();
}
void RTLib::Test::TestGLFWGLADImGuiAppFreeDelegate::FreeImGui()noexcept
{
    if (!GetParent()){
        return;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtendedData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWGLADImGuiAppExtendedData*>(app->GetExtendedData());
    appExtData->FreeImGui();
}
