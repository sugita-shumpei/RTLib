#ifndef RTLIB_EXT_OPX7_OPX7_EXCEPTIONS_H
#define RTLIB_EXT_OPX7_OPX7_EXCEPTIONS_H
#include <RTLib/Core/Exceptions.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <string>
#define RTLIB_EXT_OPX7_THROW_IF_FAILED(EXPR) \
do { \
	auto result = EXPR;\
	if (result != OPTIX_SUCCESS) { throw RTLib::Ext::OPX7::OPX7Exception(__FILE__,__LINE__,result);} \
}while(0)
#ifndef NDEBUG
#define RTLIB_EXT_OPX7_THROW_IF_FAILED_DEBUG(EXPR) \
RTLIB_EXT_OPX7_THROW_IF_FAILED(EXPR)
#else
#define RTLIB_EXT_OPX7_THROW_IF_FAILED_DEBUG(EXPR) \
do { \
	auto result = EXPR; \
}while(0)
#endif
#define RTLIB_EXT_OPX7_THROW_IF_FAILED_WITH_LOG(EXPR, LOG) \
do { \
	auto result = EXPR;\
	if (result != OPTIX_SUCCESS) { throw RTLib::Ext::OPX7::OPX7Exception(__FILE__,__LINE__,result,LOG);} \
}while(0)
#ifndef NDEBUG
#define RTLIB_EXT_OPX7_THROW_IF_FAILED_WITH_LOG_DEBUG(EXPR, LOG) \
RTLIB_EXT_OPX7_THROW_IF_FAILED_WITH_LOG(EXPR,LOG)
#else
#define RTLIB_EXT_OPX7_THROW_IF_FAILED_WITH_LOG_DEBUG(EXPR, LOG) \
do { \
	auto result = EXPR; \
}while(0)
#endif
namespace RTLib
{
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Exception : public Core::Exception {
				RTLIB_CORE_EXCEPTION_DECLARE_DERIVED_METHOD(OPX7Exception, OPX7Exception);
			public:
				OPX7Exception(const char* filename, uint32_t line, OptixResult result)noexcept
					:OPX7Exception(filename, line, ResultToString(result)) {}
				OPX7Exception(const char* filename, uint32_t line, OptixResult result, const char* log)noexcept
					:OPX7Exception(filename, line, ResultToString(result)+std::string("\nLog: ")+std::string(log)) {}
			private:
				static auto ResultToString(OptixResult result)noexcept->std::string;
			};
		}
	}
}
#endif
