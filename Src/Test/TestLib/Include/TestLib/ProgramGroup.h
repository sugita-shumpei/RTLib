#ifndef TEST_TESTLIB_PROGRAM_GROUP__H
#define TEST_TESTLIB_PROGRAM_GROUP__H
#include <optix.h>
#include <string>
namespace TestLib
{
	struct Context;
	struct Module;
	struct PipelineGroup;
	struct ProgramGroup
	{
		ProgramGroup(std::string name, PipelineGroup* pipelineGroup, OptixProgramGroupKind kind);
		virtual ~ProgramGroup();

		ProgramGroup(const ProgramGroup&) = delete;
		ProgramGroup& operator=(const ProgramGroup&) = delete;

		auto get_name() const noexcept -> std::string;

		auto get_kind() const noexcept -> OptixProgramGroupKind;

		auto get_context() const noexcept -> Context*;

		auto get_pipeline_group() const noexcept -> PipelineGroup*;

		virtual auto get_opx7_program_group() const noexcept -> OptixProgramGroup = 0;
		virtual auto get_stack_sizes() const noexcept -> OptixStackSizes = 0;

	private:
		PipelineGroup* m_PipelineGroup;
		OptixProgramGroupKind m_Kind;
		std::string m_Name;
	};
	struct ProgramGroupRaygen :public ProgramGroup
	{
		ProgramGroupRaygen(std::string name, PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName);

		virtual ~ProgramGroupRaygen();

		virtual auto get_opx7_program_group() const noexcept -> OptixProgramGroup override { return m_Opx7ProgramGroup; }
		virtual auto get_stack_sizes() const noexcept -> OptixStackSizes override { return m_StackSizes; }
		auto get_entry_function_name() const noexcept -> const char* { return m_EntryFunctionName.c_str(); }

		auto get_module() const->Module* { return m_Opx7Module; }
	private:
		OptixProgramGroup m_Opx7ProgramGroup;
		Module* m_Opx7Module;
		std::string m_EntryFunctionName;
		OptixStackSizes m_StackSizes;
	};
	struct ProgramGroupMiss :public ProgramGroup
	{
		ProgramGroupMiss(std::string name, PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName);

		virtual ~ProgramGroupMiss();

		virtual auto get_opx7_program_group() const noexcept -> OptixProgramGroup override { return m_Opx7ProgramGroup; }
		virtual auto get_stack_sizes() const noexcept -> OptixStackSizes override { return m_StackSizes; }
		auto get_entry_function_name() const noexcept -> const char* { return m_EntryFunctionName.c_str(); }

		auto get_module() const->Module* { return m_Opx7Module; }
	private:
		OptixProgramGroup m_Opx7ProgramGroup;
		Module* m_Opx7Module;
		std::string m_EntryFunctionName;
		OptixStackSizes m_StackSizes;
	};
	struct ProgramGroupHitgroup :public ProgramGroup
	{
		ProgramGroupHitgroup(std::string name, PipelineGroup* pipelineGroup,
			Module* moduleCh, std::string entryFunctionNameCh,
			Module* moduleAh, std::string entryFunctionNameAh,
			Module* moduleIs, std::string entryFunctionNameIs
		);

		virtual ~ProgramGroupHitgroup();

		virtual auto get_opx7_program_group() const noexcept -> OptixProgramGroup override { return m_Opx7ProgramGroup; }
		virtual auto get_stack_sizes() const noexcept -> OptixStackSizes override { return m_StackSizes; }

		auto get_entry_function_name_ch() const noexcept -> const char* { return m_EntryFunctionNameCH.c_str(); }
		auto get_entry_function_name_ah() const noexcept -> const char* { return m_EntryFunctionNameAH.c_str(); }
		auto get_entry_function_name_is() const noexcept -> const char* { return m_EntryFunctionNameIS.c_str(); }

		auto get_module_ch() const->Module* { return m_ModuleCH; }
		auto get_module_ah() const->Module* { return m_ModuleAH; }
		auto get_module_is() const->Module* { return m_ModuleIS; }
	private:
		OptixProgramGroup m_Opx7ProgramGroup;

		Module* m_ModuleCH;
		Module* m_ModuleAH;
		Module* m_ModuleIS;

		std::string m_EntryFunctionNameCH;
		std::string m_EntryFunctionNameAH;
		std::string m_EntryFunctionNameIS;
		OptixStackSizes m_StackSizes;
	};
	struct ProgramGroupCallable : public ProgramGroup
	{
		ProgramGroupCallable(std::string name, PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName, bool supportTrace = false);
		virtual ~ProgramGroupCallable() noexcept;

		virtual auto get_opx7_program_group() const noexcept -> OptixProgramGroup override { return m_Opx7ProgramGroup; }
		virtual auto get_stack_sizes() const noexcept -> OptixStackSizes override { return m_StackSizes; }
		auto get_entry_function_name() const noexcept -> const char* { return m_EntryFunctionName.c_str(); }

		auto get_module() const->Module* { return m_Opx7Module; }

		bool support_trace() const noexcept { return m_SupportTrace; }
	private:
		OptixProgramGroup m_Opx7ProgramGroup;
		Module* m_Opx7Module;
		std::string m_EntryFunctionName;
		bool m_SupportTrace;
		OptixStackSizes m_StackSizes;
	};
}
#endif
