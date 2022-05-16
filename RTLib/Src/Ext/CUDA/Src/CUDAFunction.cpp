#include <RTLib/Ext/CUDA/CUDAFunction.h>
#include <RTLib/Ext/CUDA/CUDAModule.h>
#include <iostream>
#include <string>

auto RTLib::Ext::CUDA::CUDAFunction::Load(CUDAModule* cuModule, const char* entryPoint) -> CUDAFunction*
{
    if (!cuModule) { return nullptr; }
    CUfunction function = nullptr;
    auto result = cuModuleGetFunction(&function, cuModule->GetCUModule(), entryPoint);
    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return nullptr;
    }
    return new CUDAFunction(cuModule,function);
}

RTLib::Ext::CUDA::CUDAFunction::~CUDAFunction() noexcept
{
    m_Function = nullptr;
    m_Module   = nullptr;
}

void RTLib::Ext::CUDA::CUDAFunction::Destory() noexcept
{
    m_Module   = nullptr;
    m_Function = nullptr;
}

bool RTLib::Ext::CUDA::CUDAFunction::Launch(const CUDAKernelLaunchDesc& desc)
{
	if (!m_Function) { return false; }
	auto pStream = (CUstream)nullptr;
	auto pKernelParams = desc.kernelParams;
	if (desc.stream) {
		pStream = desc.stream->GetCUStream();
	}
	auto result = cuLaunchKernel(
		m_Function,
		desc.gridDimX,
		desc.gridDimY,
		desc.gridDimZ,
		desc.blockDimX,
		desc.blockDimY,
		desc.blockDimZ,
		desc.sharedMemBytes,
		pStream,
		pKernelParams.data(),
		nullptr);
	if (result != CUDA_SUCCESS) {
		const char* errString = nullptr;
		(void)cuGetErrorString(result, &errString);
		std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
		return false;
	}
	return true;
}

RTLib::Ext::CUDA::CUDAFunction::CUDAFunction(CUDAModule* module, CUfunction function) noexcept:m_Module{module},m_Function{function}
{
}

auto RTLib::Ext::CUDA::CUDAFunction::GetCUFunction() noexcept -> CUfunction
{
    return m_Function;
}
