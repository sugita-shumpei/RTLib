#include <Test1_Pipeline.h>
#include <Test1_PipelineGroup.h>
#include <Test1_ProgramGroup.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>


Test1::Pipeline::Pipeline(
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
m_Opx7PipelineLinkOptions{linkOptions}
{
    auto context = m_PipelineGroup->get_context();
    auto pipeline = context->get_opx7_device_context();
    auto opx7ProgramGroups = std::vector<OptixProgramGroup>();
    
    {
        if (programGroupRg){
            opx7ProgramGroups.push_back(programGroupRg->get_opx7_program_group());
        }
    }

    for (auto programGroupMs : m_ProgramGroupsMs) 
    {
        if (programGroupMs) {
            opx7ProgramGroups.push_back(programGroupMs->get_opx7_program_group());
        }
    }
    for (auto programGroupHg : m_ProgramGroupsHg)
    {
        if (programGroupHg) {
            opx7ProgramGroups.push_back(programGroupHg->get_opx7_program_group());
        }
    }
    for (auto programGroupDc : m_ProgramGroupsDc)
    {
        if (programGroupDc) {
            opx7ProgramGroups.push_back(programGroupDc->get_opx7_program_group());
        }
    }
    for (auto programGroupCc : m_ProgramGroupsCc)
    {
        if (programGroupCc) {
            opx7ProgramGroups.push_back(programGroupCc->get_opx7_program_group());
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

Test1::Pipeline::~Pipeline()
{
    OTK_ERROR_CHECK(optixPipelineDestroy(m_Opx7Pipeline));
    m_Opx7Pipeline = nullptr;
}

auto Test1::Pipeline::get_name() const noexcept -> const char*
{
    return m_Name.c_str();
}

auto Test1::Pipeline::get_context() const noexcept ->const Context*
{
    if (!m_PipelineGroup) { return nullptr; }
    return m_PipelineGroup->get_context();
}

auto Test1::Pipeline::get_pipeline_group() const noexcept -> PipelineGroup*
{
    return m_PipelineGroup;
}

auto Test1::Pipeline::get_opx7_pipeline() const noexcept -> OptixPipeline
{
    return m_Opx7Pipeline;
}

auto Test1::Pipeline::get_opx7_pipeline_link_options() const noexcept -> const OptixPipelineLinkOptions&
{
    // TODO: return ステートメントをここに挿入します
    return m_Opx7PipelineLinkOptions;
}


auto Test1::Pipeline::get_program_groups() const noexcept ->std::vector<ProgramGroup*>
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

void Test1::Pipeline::launch(CUstream stream, CUdeviceptr params, size_t paramsSize, const ShaderBindingTable* sbt, unsigned int width, unsigned int height, unsigned int depth)
{
    auto opx7_shader_binding_table = sbt->get_opx7_shader_binding_table();
    OTK_ERROR_CHECK(optixLaunch(m_Opx7Pipeline, stream, params,paramsSize, &opx7_shader_binding_table, width, height, depth));
}
