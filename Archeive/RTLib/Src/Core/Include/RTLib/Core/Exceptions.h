#ifndef RTLIB_CORE_EXCEPTIONS_H
#define RTLIB_CORE_EXCEPTIONS_H
#include <stdexcept>
#include <string>
#include <cstdint>
#ifndef NDEBUG
#define RTLIB_CORE_ASSERT_IF_FAILED(EXPR) do {\
		\
		auto result = EXPR; \
		assert(result); \
} while (0)
#else
#define RTLIB_CORE_ASSERT_IF_FAILED(EXPR) EXPR
#endif
#define RTLIB_CORE_EXCEPTION_DECLARE_DERIVED_METHOD(EXCEPTION, NAME) \
private: \
	static inline constexpr char exceptionName[] = #NAME; \
public : \
	EXCEPTION(const char*        filename,  \
              uint32_t           line, \
			  const std::string& message)noexcept : RTLib::Core::Exception(exceptionName, filename, line, message){} \
	virtual ~EXCEPTION()noexcept {}

    
namespace RTLib
{
	namespace Core
	{
		class Exception : public std::runtime_error
		{
		public:
			Exception(const char*        classname,
				      const char*        filename, 
				      uint32_t           line,
				      const std::string& message)noexcept :std::runtime_error(FormatMessage(classname,filename,line, message)) {}
			virtual ~Exception()noexcept {}
		private:
			static auto FormatMessage(
				const char* classname,
				const char* filename,
				uint32_t           line,
				const std::string& message)noexcept -> std::string;
		};
	}
}
#endif
