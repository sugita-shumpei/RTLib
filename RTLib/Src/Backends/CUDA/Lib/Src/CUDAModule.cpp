#include <RTLib/Backends/CUDA/CUDAModule.h>
#include <RTLib/Backends/CUDA/CUDAFunction.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include <RTLib/Backends/CUDA/CUDAFunction.h>
#include "CUDAInternals.h"
struct RTLib::Backends::Cuda::Module::Impl {
    Impl(const char* filename) noexcept {
        assert(CurrentContext::Handle().Get());
        RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuModuleLoad(&module, filename));
    }
    Impl(const std::vector<char>& image) noexcept {
        assert(CurrentContext::Handle().Get());
        RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuModuleLoadData(&module, image.data()));
    }
    Impl(const std::vector<char>& image, const std::unordered_map<JitOption, void*>& options) noexcept {
        assert(CurrentContext::Handle().Get());
        unsigned int numOptions = options.size();
        auto tmpKeys = std::vector<CUjit_option>();
        auto tmpVals = std::vector<void*>();
        for (auto& [key,val] : options) {
            auto jitOptions = Internals::GetCUjit_option(key);
            tmpKeys.push_back(jitOptions);
            tmpVals.push_back(val);
        }
        RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuModuleLoadDataEx(&module, image.data(), numOptions,tmpKeys.data(),tmpVals.data()));
    }
    ~Impl()noexcept {
        assert(CurrentContext::Handle().Get());
        RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuModuleUnload(module));
    }
    CUmodule module;
    std::unordered_map<std::string, std::unique_ptr<Function>> functions = {};
};

RTLib::Backends::Cuda::Module::Module(const char* filename) noexcept
    :m_Impl{ new Impl(filename) } {}

RTLib::Backends::Cuda::Module::Module(const std::vector<char>& image) noexcept
    :m_Impl{ new Impl(image) }
{}

RTLib::Backends::Cuda::Module::Module(const std::vector<char>& image, const std::unordered_map<JitOption, void*>& options) noexcept
    :m_Impl{new Impl(image,options)}
{
}

RTLib::Backends::Cuda::Module::~Module() noexcept
{
    m_Impl.reset();
}

auto RTLib::Backends::Cuda::Module::GetHandle() const noexcept -> void*
{
    return m_Impl->module;
}

auto RTLib::Backends::Cuda::Module::GetFunction(const char* name) noexcept -> Function*
{
    if (m_Impl->functions.count(name) > 0) {
        return m_Impl->functions.at(name).get();
    }
    else {
        CUfunction tmpHandle = nullptr;
        RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuModuleGetFunction(&tmpHandle, m_Impl->module, name));
        auto ptr = new Function(tmpHandle);
        m_Impl->functions[name] = std::unique_ptr<Function>(ptr);
        return ptr;
    }
}

auto RTLib::Backends::Cuda::Module::GetGlobal(const char* name) const noexcept -> GlobalAddressDesc
{

    assert(CurrentContext::Handle().Get());
    GlobalAddressDesc desc;
    CUdeviceptr tmpPtr;
    RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuModuleGetGlobal(&tmpPtr,&desc.bytes,m_Impl->module,name));
    desc.pointer = reinterpret_cast<void*>(tmpPtr);
    return desc;
}
