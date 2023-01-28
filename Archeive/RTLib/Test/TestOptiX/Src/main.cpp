#include <TestOptiX.h>
int main(int argc, const char** argv)
{
	auto app = std::make_unique<RTLib::Test::TestOptiXApplication>();
	return app->Run(argc,argv);
}