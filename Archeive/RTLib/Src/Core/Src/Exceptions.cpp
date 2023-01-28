#include <RTLib/Core/Exceptions.h>
auto RTLib::Core::Exception::FormatMessage(
	const char*        classname,
	const char*        filename,
	uint32_t           line,
	const std::string& message)noexcept -> std::string
{
	return std::string("RTLib Exception \'") + std::string(classname) + std::string("\' Throw In File \'") + std::string(filename) + std::string("\' Line ") + std::to_string(line) + std::string("\n") + message;
}