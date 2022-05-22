#include <RTLib/Ext/CUDA/CUDAExceptions.h>
auto RTLib::Ext::CUDA::CUDAException::ResultToString(CUresult result) noexcept -> std::string {
	const char* err = nullptr;
	(void)cuGetErrorString(result, &err);
	return "Error: " + std::string(err);
}
