#include <RTLibCoreTestConfig.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Core/BinaryReader.h>
#include <RTLib/Core/Exceptions.h>
#include <iostream>
#include <fstream>
class TestException : public RTLib::Core::Exception
{
public:
	RTLIB_CORE_EXCEPTION_DECLARE_DERIVED_METHOD(TestException, TestException);
};
int main() {
	try {
		throw TestException(__FILE__, __LINE__,"Test");
	}
	catch (TestException& err) {
		std::cerr << err.what() << std::endl;
	}
	auto objModelAssetLoader = RTLib::Core::ObjModelAssetManager(".");
	RTLIB_CORE_ASSERT_IF_FAILED(objModelAssetLoader.LoadAsset("CornellBox-Original", RTLIB_CORE_TEST_CONFIG_DATA_PATH"/Models/CornellBox/CornellBox-Original.obj"));


	return 0;
}