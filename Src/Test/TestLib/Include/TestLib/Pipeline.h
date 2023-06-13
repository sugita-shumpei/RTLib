#ifndef TEST_TESTLIB_PIPELINE__H
#define TEST_TESTLIB_PIPELINE__H
#include <TestLib/ShaderBindingTable.h>
#include <optix.h>
#include <vector>
#include <string>
namespace TestLib
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

		auto get_max_stack_sizes() const noexcept -> OptixStackSizes;

		auto get_max_trace_depth()const noexcept -> unsigned int;

		void set_direct_callable_stack_size_from_traversal(unsigned int directCallableStackSizeFromTraversal) noexcept;
		auto get_direct_callable_stack_size_from_traversal() const noexcept -> unsigned int;

		void set_direct_callable_stack_size_from_state(unsigned int directCallableStackSizeFromState) noexcept;
		auto get_direct_callable_stack_size_from_state() const noexcept -> unsigned int;

		void set_continuation_stack_size(unsigned int continuationStackSize) noexcept;
		auto get_continuation_stack_size() const noexcept -> unsigned int;

		void set_max_traversable_graph_depth(unsigned int maxTraversableGraphDepth) noexcept;
		auto get_max_traversable_graph_depth() const noexcept -> unsigned int;

		void compute_stack_sizes(unsigned int maxCCDepth, unsigned int maxDCDepth);

		void launch(CUstream stream, CUdeviceptr params, size_t paramsSize, const ShaderBindingTable* sbt, unsigned int width, unsigned int height, unsigned int depth);
		
		void update();

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
		OptixStackSizes m_MaxStackSizes;
		unsigned int m_MaxTraceDepth;
		unsigned int m_DirectCallableStackSizeFromTraversal;
		unsigned int m_DirectCallableStackSizeFromState;
		unsigned int m_ContinuationStackSize;
		unsigned int m_MaxTraversableGraphDepth;
	};
}
#endif
