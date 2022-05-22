#ifndef RTLIB_EXT_CUDA_CUDA_EXCEPTIONS_H
#define RTLIB_EXT_CUDA_CUDA_EXCEPTIONS_H
#include <RTLib/Core/Exceptions.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <string>
#define RTLIB_EXT_CUDA_THROW_IF_FAILED(EXPR) \
do { \
	auto result = EXPR;\
	if (result != CUDA_SUCCESS) { throw RTLib::Ext::CUDA::CUDAException(__FILE__,__LINE__,result);} \
}while(0)
#ifndef NDEBUG
#define RTLIB_EXT_CUDA_THROW_IF_FAILED_DEBUG(EXPR) \
do { \
	auto result = EXPR;\
	if (result != CUDA_SUCCESS) { throw RTLib::Ext::CUDA::CUDAException(__FILE__,__LINE__,result);} \
}while(0)
#else
#define RTLIB_EXT_CUDA_THROW_IF_FAILED_DEBUG(EXPR) \
do { \
	auto result = EXPR; \
}while(0)
#endif
namespace RTLib
{
	namespace Ext
	{
		namespace CUDA
		{
			class CUDAException : public Core::Exception {
				RTLIB_CORE_EXCEPTION_DECLARE_DERIVED_METHOD(CUDAException, CUDAException);
			public:
				CUDAException(const char* filename, uint32_t line, CUresult result)noexcept 
				:CUDAException(filename,line,ResultToString(result)) {}
			private:
				static auto ResultToString(CUresult result)noexcept->std::string;
			};
		}
	}
}
#endif
