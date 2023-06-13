#include <TestLib/Pipeline.h>
#include <TestLib/PipelineGroup.h>
#include <TestLib/ProgramGroup.h>
#include <TestLib/Utils.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>


TestLib::Pipeline::Pipeline(
    std::string name, 
    PipelineGroup* pipelineGroup,
    ProgramGroup* programGroupRg,
    const std::vector<ProgramGroup*>& programGroupsMs, 
    const std::vector<ProgramGroup*>& programGroupsHg,
    const std::vector<ProgramGroup*>& programGroupsDc,
    const std::vector<ProgramGroup*>& programGroupsCc,
    const OptixPipelineLinkOptions& linkOptions)
:
m_Name{name}, m_PipelineGroup{pipelineGroup},
m_ProgramGroupRg{programGroupRg},
m_ProgramGroupsMs{programGroupsMs},
m_ProgramGroupsHg{programGroupsHg},
m_ProgramGroupsDc{programGroupsDc},
m_ProgramGroupsCc{programGroupsCc},
m_Opx7PipelineLinkOptions{linkOptions},
m_MaxStackSizes{}
{
    auto context = m_PipelineGroup->get_context();
    auto pipeline = context->get_opx7_device_context();
    auto opx7ProgramGroups = std::vector<OptixProgramGroup>();
    
    {
        if (programGroupRg){
            opx7ProgramGroups.push_back(programGroupRg->get_opx7_program_group());
            m_MaxStackSizes = max(m_MaxStackSizes, programGroupRg->get_stack_sizes());
        }
    }

    for (auto programGroupMs : m_ProgramGroupsMs) 
    {
        if (programGroupMs) {
            opx7ProgramGroups.push_back(programGroupMs->get_opx7_program_group());
            m_MaxStackSizes = max(m_MaxStackSizes, programGroupMs->get_stack_sizes());
        }
    }
    for (auto programGroupHg : m_ProgramGroupsHg)
    {
        if (programGroupHg) {
            opx7ProgramGroups.push_back(programGroupHg->get_opx7_program_group());
            m_MaxStackSizes = max(m_MaxStackSizes, programGroupHg->get_stack_sizes());
        }
    }
    for (auto programGroupDc : m_ProgramGroupsDc)
    {
        if (programGroupDc) {
            opx7ProgramGroups.push_back(programGroupDc->get_opx7_program_group());
            m_MaxStackSizes = max(m_MaxStackSizes, programGroupDc->get_stack_sizes());
        }
    }
    for (auto programGroupCc : m_ProgramGroupsCc)
    {
        if (programGroupCc) {
            opx7ProgramGroups.push_back(programGroupCc->get_opx7_program_group());
            m_MaxStackSizes = max(m_MaxStackSizes, programGroupCc->get_stack_sizes());
        }
    }

    auto& opx7PipelineCompileOptions = m_PipelineGroup->get_options();
    auto& opx7PipelineLinkOptions = m_Opx7PipelineLinkOptions;

    char log[1024];
    size_t logSize = sizeof(log);
    OTK_ERROR_CHECK(optixPipelineCreate(
        context->get_opx7_device_context(),
        &opx7PipelineCompileOptions,
        &opx7PipelineLinkOptions,
        opx7ProgramGroups.data(),
        opx7ProgramGroups.size(),
        log, &logSize,
        &m_Opx7Pipeline
    ));

}

TestLib::Pipeline::~Pipeline()
{
    OTK_ERROR_CHECK(optixPipelineDestroy(m_Opx7Pipeline));
    m_Opx7Pipeline = nullptr;
}

auto TestLib::Pipeline::get_name() const noexcept -> const char*
{
    return m_Name.c_str();
}

auto TestLib::Pipeline::get_context() const noexcept ->const Context*
{
    if (!m_PipelineGroup) { return nullptr; }
    return m_PipelineGroup->get_context();
}

auto TestLib::Pipeline::get_pipeline_group() const noexcept -> PipelineGroup*
{
    return m_PipelineGroup;
}

auto TestLib::Pipeline::get_opx7_pipeline() const noexcept -> OptixPipeline
{
    return m_Opx7Pipeline;
}

auto TestLib::Pipeline::get_opx7_pipeline_link_options() const noexcept -> const OptixPipelineLinkOptions&
{
    // TODO: return ステートメントをここに挿入します
    return m_Opx7PipelineLinkOptions;
}


auto TestLib::Pipeline::get_program_groups() const noexcept ->std::vector<ProgramGroup*>
{
    std::vector<ProgramGroup*> programGroups(1 + m_ProgramGroupsMs.size() + m_ProgramGroupsHg.size() + m_ProgramGroupsCc.size() + m_ProgramGroupsDc.size());
    uint32_t index = 0;
    programGroups[index] = m_ProgramGroupRg; ++index;
    for (auto programGroup : m_ProgramGroupsMs) { programGroups[index] = programGroup; ++index; }
    for (auto programGroup : m_ProgramGroupsHg) { programGroups[index] = programGroup; ++index; }
    for (auto programGroup : m_ProgramGroupsCc) { programGroups[index] = programGroup; ++index; }
    for (auto programGroup : m_ProgramGroupsDc) { programGroups[index] = programGroup; ++index; }
    return programGroups;
}

auto TestLib::Pipeline::get_max_stack_sizes() const noexcept -> OptixStackSizes
{
    return m_MaxStackSizes;
}

auto TestLib::Pipeline::get_max_trace_depth() const noexcept -> unsigned int
{
    return m_Opx7PipelineLinkOptions.maxTraceDepth;
}

void TestLib::Pipeline::launch(CUstream stream, CUdeviceptr params, size_t paramsSize, const ShaderBindingTable* sbt, unsigned int width, unsigned int height, unsigned int depth)
{
    auto opx7_shader_binding_table = sbt->get_opx7_shader_binding_table();
    OTK_ERROR_CHECK(optixLaunch(m_Opx7Pipeline, stream, params,paramsSize, &opx7_shader_binding_table, width, height, depth));
}

void TestLib::Pipeline::set_direct_callable_stack_size_from_traversal(unsigned int directCallableStackSizeFromTraversal) noexcept
{
    m_DirectCallableStackSizeFromTraversal = directCallableStackSizeFromTraversal;
}

auto TestLib::Pipeline::get_direct_callable_stack_size_from_traversal() const noexcept -> unsigned int
{
    return m_DirectCallableStackSizeFromTraversal;
}

void TestLib::Pipeline::set_direct_callable_stack_size_from_state(unsigned int directCallableStackSizeFromState) noexcept
{
    m_DirectCallableStackSizeFromState = directCallableStackSizeFromState;
}

auto TestLib::Pipeline::get_direct_callable_stack_size_from_state() const noexcept -> unsigned int
{
    return m_DirectCallableStackSizeFromState;
}

void TestLib::Pipeline::set_continuation_stack_size(unsigned int continuationStackSize) noexcept
{
    m_ContinuationStackSize = continuationStackSize;
}

auto TestLib::Pipeline::get_continuation_stack_size() const noexcept -> unsigned int
{
    return m_ContinuationStackSize;
}

void TestLib::Pipeline::set_max_traversable_graph_depth(unsigned int maxTraversableGraphDepth) noexcept
{
    m_MaxTraversableGraphDepth = maxTraversableGraphDepth;
}

auto TestLib::Pipeline::get_max_traversable_graph_depth() const noexcept -> unsigned int
{
    return m_MaxTraversableGraphDepth;
}

void TestLib::Pipeline::compute_stack_sizes(unsigned int maxCCDepth, unsigned int maxDCDepth)
{
    OTK_ERROR_CHECK(optixUtilComputeStackSizes(
        &m_MaxStackSizes,
        m_Opx7PipelineLinkOptions.maxTraceDepth,
        maxCCDepth,
        maxDCDepth,
        &m_DirectCallableStackSizeFromTraversal,
        &m_DirectCallableStackSizeFromState,
        &m_ContinuationStackSize)
    );
}

void TestLib::Pipeline::update()
{
    OTK_ERROR_CHECK(optixPipelineSetStackSize(
        m_Opx7Pipeline,
        m_DirectCallableStackSizeFromTraversal,
        m_DirectCallableStackSizeFromState,
        m_ContinuationStackSize,
        m_MaxTraversableGraphDepth)
    );
}
