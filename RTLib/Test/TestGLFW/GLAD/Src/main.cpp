#include <TestGLFW_GLAD.h>
int main(int argc, const char** argv)
{
    auto app = std::make_unique<RTLib::Test::TestGLFWApplication>();
    app->AddInitDelegate<RTLib::Test::TestGLFWGLADAppInitDelegate>(256, 256, "title");
    app->AddMainDelegate<RTLib::Test::TestGLFWGLADAppMainDelegate>();
    return app->Run(argc,argv);
}
