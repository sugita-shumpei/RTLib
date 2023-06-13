#ifndef TEST_TESTLIB_MODULE__H
#define TEST_TESTLIB_MODULE__H
#include <TestLib/PipelineGroup.h>
#include <iostream>
#include <vector>
#include <optix.h>
#include <cuda.h>
namespace TestLib
{
	//struct ModuleBoundValueEntry
	//{
	//	ModuleBoundValueEntry();
	//	
	//	ModuleBoundValueEntry(const ModuleBoundValueEntry&) = default;
	//	ModuleBoundValueEntry& operator=(const ModuleBoundValueEntry&) = default;

	//	auto get_pipeline_params_offset_in_bytes() const noexcept -> std::size_t;
	//	void set_pipeline_params_offset_in_bytes(std::size_t offset) noexcept;

	//	auto get_size_in_bytes() const noexcept -> std::size_t;
	//	void set_size_in_bytes(std::size_t sizeInBytes) noexcept;

	//	auto get_annotation() const noexcept -> const char*;
	//	void set_annotation(std::string annotation)noexcept;

	//	auto get_data() const noexcept -> void*;

	//	template<typename T>
	//	void set_data(T* ptr);
	//private: 
	//	std::size_t           m_PipelineParamOffsetInBytes = 0;
	//	std::size_t           m_SizeInBytes = 0;
	//	std::shared_ptr<void> m_Ptr = nullptr;
	//	std::string           m_Annotation = "";
	//};
	struct ModuleOptions
	{
		int                                   maxRegisterCount;
		OptixCompileOptimizationLevel         optLevel;
		OptixCompileDebugLevel                debugLevel;
		//std::vector< ModuleBoundValueEntry> boundValues;
		std::vector<OptixPayloadType>         payloadTypes;

		auto get_opx7_module_compile_options() noexcept -> OptixModuleCompileOptions
		{
			OptixModuleCompileOptions ops = {};
			ops.maxRegisterCount = maxRegisterCount;
			ops.optLevel         = optLevel;
			ops.debugLevel       = debugLevel;
			ops.numPayloadTypes  = payloadTypes.size();
			ops.payloadTypes     = payloadTypes.data();
			return ops;
		}
	};

	struct Module
	{
		 Module(std::string name, TestLib::PipelineGroup* pipelineGroup, const ModuleOptions& options, const char* ptx, size_t ptxSize);
		 Module(std::string name, TestLib::PipelineGroup* pipelineGroup, const ModuleOptions& options, const OptixBuiltinISOptions& builtinIsOptions);
		~Module();

		Module(const Module&) = delete;
		Module& operator=(const Module&) = delete;

		auto get_name() const noexcept -> const char*;

		auto get_pipeline_group() const noexcept -> const PipelineGroup*;

		auto get_pipeline_group() noexcept -> PipelineGroup*;

		auto get_context() const noexcept -> const Context*;

		auto get_options() const noexcept -> const ModuleOptions&;

		auto get_opx7_module() const noexcept -> OptixModule;

	private:
		TestLib::PipelineGroup* m_PipelineGroup;
		OptixModule m_Opx7Module;
		ModuleOptions m_Options;
		std::string m_Name;
	};
}
#endif
