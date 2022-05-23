#include <RTLib/Ext/OPX7/OPX7Pipeline.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7ProgramGroup.h>
#include <RTLib/Ext/OPX7/OPX7Exceptions.h>
#include <optix_stubs.h>
#include <cassert>
struct RTLib::Ext::OPX7::OPX7Pipeline::Impl
{
    Impl(const OPX7PipelineCreateDesc& desc)noexcept :
        compileOptions{ desc.compileOptions },
        linkOptions{ desc.linkOptions }
    {}
    void SetContext(OPX7Context* ctx)noexcept {
        if (!ctx) { return; }
        context = ctx;
    }
    OPX7Context*               context  = nullptr;
    OptixPipeline              opxPipeline       = nullptr;
    OPX7PipelineCompileOptions compileOptions = {};
    OPX7PipelineLinkOptions    linkOptions    = {};
};
auto RTLib::Ext::OPX7::OPX7Pipeline::New(OPX7Context* context, const OPX7PipelineCreateDesc& desc) -> OPX7Pipeline*
{
    if (!context) { return nullptr; }
    auto pipeline = new OPX7Pipeline(desc);
    auto opxProgramGroups = std::vector<OptixProgramGroup>(desc.programGroups.size(), nullptr);
    auto programGroups = desc.programGroups;
    for (size_t i = 0; i < opxProgramGroups.size(); ++i) {
        assert(programGroups[i] != nullptr);
        opxProgramGroups[i] = programGroups[i]->GetOptixProgramGroup();
    }
    auto deviceContext = context->GetOptixDeviceContext();
    pipeline->m_Impl->SetContext(context);
    char logString[1024];
    size_t length = sizeof(logString);
    OptixPipelineCompileOptions compileOps = {};
    compileOps.usesMotionBlur = static_cast<int>(desc.compileOptions.usesMotionBlur);
    compileOps.usesPrimitiveTypeFlags = static_cast<unsigned int>(desc.compileOptions.usesPrimitiveTypeFlags);
    compileOps.traversableGraphFlags = static_cast<unsigned int>(desc.compileOptions.traversableGraphFlags);
    compileOps.exceptionFlags = static_cast<unsigned int>(desc.compileOptions.exceptionFlags);
    compileOps.numPayloadValues = desc.compileOptions.numPayloadValues;
    compileOps.numAttributeValues = desc.compileOptions.numAttributeValues;
    compileOps.pipelineLaunchParamsVariableName = desc.compileOptions.launchParamsVariableNames;
    OptixPipelineLinkOptions linkOps = {};
    linkOps.debugLevel    = static_cast<OptixCompileDebugLevel>(desc.linkOptions.debugLevel);
    linkOps.maxTraceDepth = desc.linkOptions.maxTraceDepth;
    auto opxPipeline = OptixPipeline(nullptr);
    RTLIB_EXT_OPX7_THROW_IF_FAILED_WITH_LOG(optixPipelineCreate(
        deviceContext,
        &compileOps,&linkOps,
        opxProgramGroups.data(),
        static_cast<unsigned int>(opxProgramGroups.size()),
        logString,&length, &opxPipeline
    ), logString);
    pipeline->m_Impl->opxPipeline = opxPipeline;
    return pipeline;
}

RTLib::Ext::OPX7::OPX7Pipeline::~OPX7Pipeline() noexcept
{
    m_Impl.reset();
}

void RTLib::Ext::OPX7::OPX7Pipeline::Destroy()
{
    if (!m_Impl) { return; }
    if (!m_Impl->opxPipeline) { return; }
    RTLIB_EXT_OPX7_THROW_IF_FAILED(optixPipelineDestroy(m_Impl->opxPipeline));
    m_Impl->opxPipeline = nullptr;
}

auto RTLib::Ext::OPX7::OPX7Pipeline::GetCompileOptions() const noexcept -> const OPX7PipelineCompileOptions&
{
    // TODO: return ステートメントをここに挿入します
    assert(m_Impl != nullptr);
    return m_Impl->compileOptions;
}

auto RTLib::Ext::OPX7::OPX7Pipeline::GetLinkOptions() const noexcept -> const OPX7PipelineLinkOptions&
{
    // TODO: return ステートメントをここに挿入します
    assert(m_Impl != nullptr);
    return m_Impl->linkOptions;
}

void RTLib::Ext::OPX7::OPX7Pipeline::Launch(CUDA::CUDAStream* stream, CUDA::CUDABufferView paramsBufferView, OPX7ShaderTable* shaderTable, unsigned int width, unsigned int height, unsigned int depth)
{
    OPX7::OPX7Context::Launch(this, stream, paramsBufferView, shaderTable, width, height, depth);
}

void RTLib::Ext::OPX7::OPX7Pipeline::Launch(CUDA::CUDABufferView paramsBufferView, OPX7ShaderTable* shaderTable, unsigned int width, unsigned int height, unsigned int depth)
{
    OPX7::OPX7Context::Launch(this, paramsBufferView, shaderTable, width, height, depth);
}

RTLib::Ext::OPX7::OPX7Pipeline::OPX7Pipeline(const OPX7PipelineCreateDesc& desc) noexcept:m_Impl{new Impl(desc)}
{
}

auto RTLib::Ext::OPX7::OPX7Pipeline::GetOptixPipeline() const noexcept -> OptixPipeline
{
    assert(m_Impl != nullptr);
    return m_Impl->opxPipeline;
}
