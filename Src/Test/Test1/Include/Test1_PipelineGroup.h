#ifndef TEST_TEST1_PIPELINE_GROUP__H
#define TEST_TEST1_PIPELINE_GROUP__H
#include <Test1_Context.h>
#include <iostream>
#include <unordered_map>
#include <optix.h>
#include <cuda.h>
namespace Test1
{
	struct Module;
	struct ModuleOptions;
	struct ProgramGroup;
	struct Pipeline;

	struct PipelineGroup
	{
		PipelineGroup(const Test1::Context* context, const OptixPipelineCompileOptions& options);
		~PipelineGroup();

		PipelineGroup(const PipelineGroup&) = delete;
		PipelineGroup& operator=(const PipelineGroup&) = delete;

		auto get_context() const noexcept -> const Test1::Context*;
		auto get_options() const noexcept -> const OptixPipelineCompileOptions&;

		bool load_module(std::string name, const ModuleOptions& options, const char* ptx, size_t ptxSize);

		auto get_module(std::string name) const noexcept -> Module*;
		bool has_module(std::string name) const noexcept;

		bool load_program_group_rg(std::string pgName, std::string moduleName  , std::string entryName);
		bool load_program_group_ms(std::string pgName, std::string moduleName  , std::string entryName);
		bool load_program_group_hg(std::string pgName, std::string moduleNameCh, std::string entryNameCh, std::string moduleNameAh, std::string entryNameAh, std::string moduleNameIs, std::string entryNameIs);
		bool load_program_group_dc(std::string pgName, std::string moduleName  , std::string entryName);
		bool load_program_group_cc(std::string pgName, std::string moduleName  , std::string entryName);

		auto get_program_group_rg(std::string pgName)const noexcept -> ProgramGroup*;
		auto get_program_group_ms(std::string pgName)const noexcept -> ProgramGroup*;
		auto get_program_group_hg(std::string pgName)const noexcept -> ProgramGroup*;
		auto get_program_group_dc(std::string pgName)const noexcept -> ProgramGroup*;
		auto get_program_group_cc(std::string pgName)const noexcept -> ProgramGroup*;

		bool has_program_group_rg(std::string pgName)const noexcept ;
		bool has_program_group_ms(std::string pgName)const noexcept ;
		bool has_program_group_hg(std::string pgName)const noexcept ;
		bool has_program_group_dc(std::string pgName)const noexcept ;
		bool has_program_group_cc(std::string pgName)const noexcept ;

		bool load_pipeline(
			std::string pipelineName,
			std::string pgNameRg,
			const std::vector<std::string>& pgNamesMs,
			const std::vector<std::string>& pgNamesHg,
			const std::vector<std::string>& pgNamesDc,
			const std::vector<std::string>& pgNamesCc,
			OptixCompileDebugLevel debugLevel,
			unsigned int maxTraceDepth
		);
		auto get_pipeline(std::string pipelineName) const noexcept -> Pipeline*;
		bool has_pipeline(std::string pipelineName) const noexcept;

	private:
		const Test1::Context* m_Context;
		OptixPipelineCompileOptions m_Options;

		std::unordered_map<std::string, std::unique_ptr<Module>>       m_Modules;
		std::unordered_map<std::string, std::unique_ptr<ProgramGroup>> m_ProgramGroupsRG;
		std::unordered_map<std::string, std::unique_ptr<ProgramGroup>> m_ProgramGroupsMS;
		std::unordered_map<std::string, std::unique_ptr<ProgramGroup>> m_ProgramGroupsHG;
		std::unordered_map<std::string, std::unique_ptr<ProgramGroup>> m_ProgramGroupsDC;
		std::unordered_map<std::string, std::unique_ptr<ProgramGroup>> m_ProgramGroupsCC;
		std::unordered_map<std::string, std::unique_ptr<Pipeline>> m_Pipelines;
	};
}
#endif
