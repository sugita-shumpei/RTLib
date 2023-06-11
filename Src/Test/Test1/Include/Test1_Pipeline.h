#ifndef TEST_TEST1_PIPELINE__H
#define TEST_TEST1_PIPELINE__H
#include <Test1_ShaderBindingTable.h>
#include <optix.h>
#include <vector>
#include <string>
namespace Test1
{
	
	struct Context;
	struct ProgramGroup;
	struct PipelineGroup;
	struct Pipeline
	{
		Pipeline(std::string name,
			PipelineGroup* pipelineGroup,
			ProgramGroup*  programGroupRg,
			const std::vector<ProgramGroup*>& programGroupsMs,
			const std::vector<ProgramGroup*>& programGroupsHg,
			const std::vector<ProgramGroup*>& programGroupsDc,
			const std::vector<ProgramGroup*>& programGroupsCc,
			const OptixPipelineLinkOptions& linkOptions
		);
		virtual ~Pipeline();

		auto get_name() const noexcept -> const char*;

		auto get_context() const noexcept ->const Context*;

		auto get_pipeline_group() const noexcept -> PipelineGroup*;
		
		auto get_opx7_pipeline() const noexcept -> OptixPipeline;

		auto get_opx7_pipeline_link_options() const noexcept -> const OptixPipelineLinkOptions& ;

		auto get_program_groups() const noexcept ->std::vector<ProgramGroup*>;

		void launch(CUstream stream, CUdeviceptr params, size_t paramsSize, const ShaderBindingTable* sbt,unsigned int width, unsigned int height, unsigned int depth);
	private:
		PipelineGroup* m_PipelineGroup;
		OptixPipeline m_Opx7Pipeline;
		OptixPipelineLinkOptions m_Opx7PipelineLinkOptions = {};
		ProgramGroup* m_ProgramGroupRg;
		std::vector<ProgramGroup*> m_ProgramGroupsMs;
		std::vector<ProgramGroup*> m_ProgramGroupsHg;
		std::vector<ProgramGroup*> m_ProgramGroupsDc;
		std::vector<ProgramGroup*> m_ProgramGroupsCc;
		std::string m_Name;
	};
}
#endif
