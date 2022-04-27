#include <TestGLFW_Imgui_GLAD.h>
#include <vector>
#include <utility>
int main(int argc, const char** argv)
{
    auto app = std::make_unique<RTLib::Test::TestGLFWApplication>();
    app->AddInitDelegate< RTLib::Test::TestGLFWImGuiGLADAppInitDelegate>(256, 256, "title");
    app->AddMainDelegate< RTLib::Test::TestGLFWImGuiGLADAppMainDelegate>();
    app->AddFreeDelegate< RTLib::Test::TestGLFWImGuiGLADAppFreeDelegate>();
    app->AddExtensionData<RTLib::Test::TestGLFWImGuiGLADAppExtensionData>();
    return app->Run(argc,argv);
}
struct RTLib::Test::TestGLFWImGuiGLADAppExtensionData::Impl{
    bool  m_IsImguiInit = false;
    float m_X           = 0.0f;
    float m_Y           = 0.0f;
};
RTLib::Test::TestGLFWImGuiGLADAppExtensionData::TestGLFWImGuiGLADAppExtensionData(TestLib::TestApplication* app)noexcept
:Test::TestGLFWAppExtensionData(static_cast<Test::TestGLFWApplication*>(app)){
    m_Impl = std::unique_ptr<RTLib::Test::TestGLFWImGuiGLADAppExtensionData::Impl>(new RTLib::Test::TestGLFWImGuiGLADAppExtensionData::Impl());
}

RTLib::Test::TestGLFWImGuiGLADAppExtensionData::~TestGLFWImGuiGLADAppExtensionData()noexcept{
    m_Impl.reset();
}

void RTLib::Test::TestGLFWImGuiGLADAppExtensionData::InitImGui(){
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
void RTLib::Test::TestGLFWImGuiGLADAppExtensionData::FreeImGui()noexcept{
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
void RTLib::Test::TestGLFWImGuiGLADAppExtensionData::DrawImGui(){
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

void RTLib::Test::TestGLFWImGuiGLADAppInitDelegate::Init()
{
    InitGLFW();
    auto windowHints = std::unordered_map<int, int>();
    windowHints[GLFW_CLIENT_API]            = GLFW_OPENGL_API;
    windowHints[GLFW_OPENGL_PROFILE]        = GLFW_OPENGL_CORE_PROFILE;
    windowHints[GLFW_OPENGL_FORWARD_COMPAT] = GLFW_TRUE;
    windowHints[GLFW_VISIBLE]                = GLFW_FALSE;
    std::vector<std::pair<int, int>> glVersions = {
        {4,6},{4,5},{4,4},{4,3},{4,2},{4,1},{4,0},
        {3,3},{3,2},{3,1},{3,0},
        {2,1},{2,0}
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
    InitGLAD();
    InitImGui();
    ShowWindow();
}
void RTLib::Test::TestGLFWImGuiGLADAppInitDelegate::InitGLAD()
{
    MakeContext();
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed To Initialize GLAD!");
    }
}
void RTLib::Test::TestGLFWImGuiGLADAppInitDelegate::InitImGui()
{
    if (!GetParent()){
        return;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtensionData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWImGuiGLADAppExtensionData*>(app->GetExtensionData());
    
    appExtData->InitImGui();
}
void RTLib::Test::TestGLFWImGuiGLADAppMainDelegate::Main() {
    while (!ShouldClose()) {
        PollEvents ();
        RenderFrame();
        RenderImGui();
        SwapBuffers();
    }
}
void RTLib::Test::TestGLFWImGuiGLADAppMainDelegate::RenderFrame() {
    int display_w, display_h;
    glfwGetFramebufferSize(GetWindow(), &display_w, &display_h);
    glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, display_w, display_h);

}
void RTLib::Test::TestGLFWImGuiGLADAppMainDelegate::RenderImGui() {
    
    if (!GetParent()){
        return;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtensionData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWImGuiGLADAppExtensionData*>(app->GetExtensionData());
    appExtData->DrawImGui();
}

void RTLib::Test::TestGLFWImGuiGLADAppFreeDelegate::Free()noexcept{
    if (!GetParent()){
        return;
    }
    FreeImGui();
    FreeWindow();
    FreeGLFW();
}
void RTLib::Test::TestGLFWImGuiGLADAppFreeDelegate::FreeImGui()noexcept
{
    if (!GetParent()){
        return;
    }
    auto app = static_cast<RTLib::Test::TestGLFWApplication*>(GetParent());
    if (!app->GetExtensionData()){
        return;
    }
    auto appExtData = static_cast<RTLib::Test::TestGLFWImGuiGLADAppExtensionData*>(app->GetExtensionData());
    appExtData->FreeImGui();
}
