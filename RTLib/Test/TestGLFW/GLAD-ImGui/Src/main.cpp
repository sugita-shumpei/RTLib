#include <TestGLFW_GLAD_Imgui.h>
int main(int argc, const char** argv)
{
    auto app = std::make_unique<RTLib::Test::TestGLFWApplication>();
    app->AddInitDelegate<RTLib::Test::TestGLFWGLADImGuiAppInitDelegate>(256, 256, "title");
    app->AddMainDelegate<RTLib::Test::TestGLFWGLADImGuiAppMainDelegate>();
    app->AddFreeDelegate<RTLib::Test::TestGLFWGLADImGuiAppFreeDelegate>();
    app->AddExtendedData<RTLib::Test::TestGLFWGLADImGuiAppExtendedData>();
    return app->Run(argc,argv);
}
