#include <TestGLFW_Common.h>
int main(int argc, const char** argv)
{
    auto app = std::make_unique<RTLib::Test::TestGLFWApplication>();
    return app->Run(argc, argv);
}
