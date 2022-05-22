#include <RTLib/Ext/OPX7/OPX7Module.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/OPX7Exceptions.h>
#include <optix_stubs.h>
#include <optix_types.h>
#include <cassert>
struct RTLib::Ext::OPX7::OPX7Module ::Impl
{
    Impl(const OPX7ModuleCreateDesc &dsc) noexcept : desc{dsc} {}
    void SetContext(OPX7Context *ctx) noexcept
    {
        context = ctx;
    }
    OPX7Context* context   = nullptr;
    OptixModule  opxModule = nullptr;
    OPX7ModuleCreateDesc desc = {};
};
auto RTLib::Ext::OPX7::OPX7Module::New(OPX7Context *context, const OPX7ModuleCreateDesc &desc) -> OPX7Module *
{
    if (!context)
    {
        return nullptr;
    }
    auto module = new OPX7Module(desc);
    if (!module)
    {
        return nullptr;
    }
    module->m_Impl->SetContext(context);
    auto deviceContext = module->m_Impl->context->GetOptixDeviceContext();
    OptixModuleCompileOptions moduleOps = {};
    auto boundValues = desc.moduleOptions.boundValueEntries;
    auto payloadTypes = desc.moduleOptions.payloadTypes;
    moduleOps.maxRegisterCount = desc.moduleOptions.maxRegisterCount;
    moduleOps.payloadTypes = payloadTypes.data();
    moduleOps.numPayloadTypes = static_cast<unsigned int>(payloadTypes.size());
    moduleOps.boundValues = boundValues.data();
    moduleOps.numBoundValues = static_cast<unsigned int>(boundValues.size());
    moduleOps.debugLevel = static_cast<OptixCompileDebugLevel>(desc.moduleOptions.debugLevel);
    moduleOps.optLevel = static_cast<OptixCompileOptimizationLevel>(desc.moduleOptions.optLevel);
    OptixPipelineCompileOptions pipelineOps = {};
    pipelineOps.usesMotionBlur = static_cast<int>(desc.pipelineOptions.usesMotionBlur);
    pipelineOps.usesPrimitiveTypeFlags = static_cast<unsigned int>(desc.pipelineOptions.usesPrimitiveTypeFlags);
    pipelineOps.traversableGraphFlags = static_cast<unsigned int>(desc.pipelineOptions.traversableGraphFlags);
    pipelineOps.exceptionFlags = static_cast<unsigned int>(desc.pipelineOptions.exceptionFlags);
    pipelineOps.numPayloadValues = desc.pipelineOptions.numPayloadValues;
    pipelineOps.numAttributeValues = desc.pipelineOptions.numAttributeValues;
    pipelineOps.pipelineLaunchParamsVariableName = desc.pipelineOptions.launchParamsVariableNames;
    OptixModule opxModule = nullptr;
    char logString[1024];
    size_t logLength = sizeof(logString);
    RTLIB_EXT_OPX7_THROW_IF_FAILED_WITH_LOG(optixModuleCreateFromPTX(deviceContext, &moduleOps, &pipelineOps, desc.ptxBinary.data(), static_cast<size_t>(desc.ptxBinary.size()), logString, &logLength, &opxModule), logString);
    module->m_Impl->opxModule = opxModule;
    return module;
}

RTLib::Ext::OPX7::OPX7Module::~OPX7Module() noexcept
{
    m_Impl.reset();
}

void RTLib::Ext::OPX7::OPX7Module::Destroy()
{
    if (!m_Impl)
    {
        return;
    }
    if (!m_Impl->opxModule)
    {
        return;
    }
    try
    {
        RTLIB_EXT_OPX7_THROW_IF_FAILED(optixModuleDestroy(m_Impl->opxModule));
    }
    catch (OPX7Exception &err)
    {
        std::cerr << err.what() << std::endl;
    }
    m_Impl->opxModule = nullptr;
}

auto RTLib::Ext::OPX7::OPX7Module::GetModuleCompileOptions() const noexcept -> const OPX7ModuleCompileOptions &
{
    assert(m_Impl != nullptr);
    return m_Impl->desc.moduleOptions;
}

auto RTLib::Ext::OPX7::OPX7Module::GetPipelineCompileOptions() const noexcept -> const OPX7PipelineCompileOptions &
{
    assert(m_Impl != nullptr);
    return m_Impl->desc.pipelineOptions;
}

RTLib::Ext::OPX7::OPX7Module::OPX7Module(const OPX7ModuleCreateDesc &desc) noexcept : m_Impl{new Impl(desc)}
{
}

auto RTLib::Ext::OPX7::OPX7Module::GetOptixModule() const noexcept -> OptixModule
{
    return (m_Impl != nullptr) ? m_Impl->opxModule : nullptr;
}
