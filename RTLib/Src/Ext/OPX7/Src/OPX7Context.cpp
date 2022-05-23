#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7Exceptions.h>
#include <RTLib/Ext/OPX7/OPX7Module.h>
#include <RTLib/Ext/OPX7/OPX7ProgramGroup.h>
#include <RTLib/Ext/OPX7/OPX7Pipeline.h>
#include <RTLib/Ext/OPX7/OPX7ShaderTable.h>
#include <RTLib/Ext/OPX7/OPX7Pipeline.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAStream.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table.h>
#include <optix_function_table_definition.h>
#include <cassert>
struct RTLib::Ext::OPX7::OPX7Context::Impl
{
    Impl(const OPX7::OPX7ContextCreateDesc& dsc)noexcept :desc{ dsc } {}
    OPX7::OPX7ContextCreateDesc desc = {};
    OptixDeviceContext deviceContext = nullptr;
};
RTLib::Ext::OPX7::OPX7Context::OPX7Context(const OPX7ContextCreateDesc& desc) noexcept :m_Impl{ new Impl(desc) } {
}

RTLib::Ext::OPX7::OPX7Context::~OPX7Context() noexcept
{

}

bool RTLib::Ext::OPX7::OPX7Context::Initialize()
{
    if (!CUDA::CUDAContext::Initialize()) {
        return false;
    }
    {
        auto result = optixInit();
        if (result != OPTIX_SUCCESS) { return false; }
    }
    {
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = m_Impl->desc.pfnLogCallback;
        options.logCallbackData     = m_Impl->desc.pCallbackData;
        options.logCallbackLevel    = static_cast<unsigned int>(m_Impl->desc.level);
        options.validationMode      = static_cast<OptixDeviceContextValidationMode>(m_Impl->desc.validationMode);
        RTLIB_EXT_OPX7_THROW_IF_FAILED(optixDeviceContextCreate(GetCUContext(), &options, &m_Impl->deviceContext));
    }
    return true;
}

void RTLib::Ext::OPX7::OPX7Context::Terminate()
{
    if (m_Impl->deviceContext) {
        try {
            RTLIB_EXT_OPX7_THROW_IF_FAILED(optixDeviceContextDestroy(m_Impl->deviceContext));
        }
        catch (OPX7Exception& err) {
            std::cerr << err.what() << std::endl;
        }
        m_Impl->deviceContext = nullptr;
    }
    CUDA::CUDAContext::Terminate();
}

auto RTLib::Ext::OPX7::OPX7Context::CreateOPXModule(const OPX7ModuleCreateDesc& desc) -> OPX7Module*
{
    return OPX7Module::New(this,desc);
}

auto RTLib::Ext::OPX7::OPX7Context::CreateOPXProgramGroups(const std::vector<OPX7ProgramGroupCreateDesc>& descs, const OPX7ProgramGroupOptions& options) -> std::vector<OPX7ProgramGroup*>
{
    return OPX7ProgramGroup::Enumerate(this,descs,options);
}

auto RTLib::Ext::OPX7::OPX7Context::CreateOPXProgramGroup(const OPX7ProgramGroupCreateDesc& desc, const OPX7ProgramGroupOptions& options) -> OPX7ProgramGroup*
{
    auto res = OPX7ProgramGroup::Enumerate(this, std::vector<OPX7ProgramGroupCreateDesc>{desc}, options);
    return res.empty() ? nullptr : res[0];
}

auto RTLib::Ext::OPX7::OPX7Context::CreateOPXPipeline(const OPX7PipelineCreateDesc& desc) -> OPX7Pipeline*
{
    return OPX7Pipeline::New(this,desc);
}

auto RTLib::Ext::OPX7::OPX7Context::CreateOPXShaderTable(const OPX7ShaderTableCreateDesc& desc) -> OPX7ShaderTable*
{
    return OPX7ShaderTable::Allocate(this,desc);
}

void RTLib::Ext::OPX7::OPX7Context::Launch(OPX7Pipeline* pipeline, CUDA::CUDAStream* stream, CUDA::CUDABufferView paramsBufferView, OPX7ShaderTable* shaderTable, unsigned int width, unsigned int height, unsigned int depth)
{
    assert(pipeline);
    auto sbt = shaderTable->GetOptixShaderBindingTable();
    RTLIB_EXT_OPX7_THROW_IF_FAILED(
        optixLaunch(pipeline->GetOptixPipeline(), GetCUstream(stream), paramsBufferView.GetDeviceAddress(), paramsBufferView.GetSizeInBytes(), &sbt, width, height, depth
    ));
}

void RTLib::Ext::OPX7::OPX7Context::Launch(OPX7Pipeline* pipeline, CUDA::CUDABufferView paramsBufferView, OPX7ShaderTable* shaderTable, unsigned int width, unsigned int height, unsigned int depth)
{
    Launch(pipeline, nullptr, paramsBufferView, shaderTable, width, height, depth);
}

auto RTLib::Ext::OPX7::OPX7Context::GetOptixDeviceContext() noexcept -> OptixDeviceContext
{
    assert(m_Impl != nullptr);
    return m_Impl->deviceContext;
}
