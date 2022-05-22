#include <RTLib/Core/BaseObject.h>
#include <RTLib/Core/Exceptions.h>
#include <iostream>
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
}