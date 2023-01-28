#include <RTLib/Ext/CUDA/CUDAModule.h>
#include <RTLib/Ext/CUDA/CUDAFunction.h>
#include <iostream>
#include <string>
auto RTLib::Ext::CUDA::CUDAModule::LoadFromFile(CUDAContext* context, const char* filename) -> CUDAModule*
{
    if (!context) { return nullptr; }
    CUmodule cuModule = nullptr;
    auto result = cuModuleLoad(&cuModule, filename);
    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return nullptr;
    }
    return new CUDAModule(context, cuModule);
}
auto RTLib::Ext::CUDA::CUDAModule::LoadFromData(CUDAContext* context, const void* data) -> CUDAModule*
{
    if (!context) { return nullptr; }
    CUmodule cuModule = nullptr;
    auto result = cuModuleLoadData(&cuModule, data);
    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return nullptr;
    }
    return new CUDAModule(context, cuModule);
}
auto RTLib::Ext::CUDA::CUDAModule::LoadFromData(CUDAContext* context, const void* data, const std::vector<CUDAJitOptionValue>& optionValues)->CUDAModule*
{
    if (!context) { return nullptr; }
    CUmodule cuModule = nullptr;
    std::vector<CUjit_option> options(optionValues.size());
    std::vector<void*>        values(optionValues.size());
    for (auto i = 0; i < optionValues.size(); ++i) {
        options[i] = static_cast<CUjit_option>(optionValues[i].option);
        values[i] = optionValues[i].value;
    }
    auto result = cuModuleLoadDataEx(&cuModule, data, optionValues.size(), options.data(), values.data());
    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return nullptr;
    }
    return new CUDAModule(context, cuModule);
}
RTLib::Ext::CUDA::CUDAModule::~CUDAModule() noexcept {}

void RTLib::Ext::CUDA::CUDAModule::Destory() noexcept
{
    if (!m_Module) {
        return;
    }
    auto result = cuModuleUnload(m_Module);
    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
    }
    m_Module = nullptr;
}

auto RTLib::Ext::CUDA::CUDAModule::LoadFunction(const char* entryPoint) -> CUDAFunction*
{
    return CUDAFunction::Load(this,entryPoint);
}

RTLib::Ext::CUDA::CUDAModule::CUDAModule(CUDAContext* context, CUmodule cuModule) noexcept :m_Context{ context }, m_Module{ cuModule } {}

auto RTLib::Ext::CUDA::CUDAModule::GetCUmodule() noexcept -> CUmodule { return m_Module; }
