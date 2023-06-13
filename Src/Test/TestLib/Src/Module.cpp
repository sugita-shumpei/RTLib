#include <TestLib/Module.h>
#include <optix_stubs.h>

TestLib::Module::Module(std::string name,TestLib::PipelineGroup* pipelineGroup, const ModuleOptions& options, const char* ptx, size_t ptxSize)
	:m_PipelineGroup{ pipelineGroup }, m_Options{ options }, m_Opx7Module{ nullptr }, m_Name{ name }
{
	auto moduleCompileOptions = m_Options.get_opx7_module_compile_options();
	auto context = m_PipelineGroup->get_context();
	auto pipelineCompileOptions = m_PipelineGroup->get_options();
	auto opx7DeviceContext = context->get_opx7_device_context();

	char log[1024];
	size_t logSize = sizeof(log);

	OTK_ERROR_CHECK(optixModuleCreateFromPTX(opx7DeviceContext, &moduleCompileOptions, &pipelineCompileOptions, ptx, ptxSize, log, &logSize, &m_Opx7Module));
	if (logSize != sizeof(log)) {
		std::cerr << log << std::endl;
	}

}

TestLib::Module::Module(std::string name, TestLib::PipelineGroup* pipelineGroup, const ModuleOptions& options, const OptixBuiltinISOptions& builtinIsOptions)
	:m_PipelineGroup{ pipelineGroup }, m_Options{ options }, m_Opx7Module{ nullptr }, m_Name{ name }
{
	auto moduleCompileOptions = m_Options.get_opx7_module_compile_options();
	auto context = m_PipelineGroup->get_context();
	auto pipelineCompileOptions = m_PipelineGroup->get_options();
	auto opx7DeviceContext = context->get_opx7_device_context();

	OTK_ERROR_CHECK(optixBuiltinISModuleGet(opx7DeviceContext, &moduleCompileOptions,&pipelineCompileOptions, &builtinIsOptions, &m_Opx7Module));
}

TestLib::Module::~Module()
{
	OTK_ERROR_CHECK(optixModuleDestroy(m_Opx7Module));
	m_Opx7Module = nullptr;
}

auto TestLib::Module::get_name() const noexcept -> const char*
{
	return m_Name.c_str();
}

auto TestLib::Module::get_pipeline_group() const noexcept -> const TestLib::PipelineGroup*
{
	return m_PipelineGroup;
}

auto TestLib::Module::get_pipeline_group() noexcept -> TestLib::PipelineGroup*
{
	return m_PipelineGroup;
}

auto TestLib::Module::get_context() const noexcept ->const TestLib::Context*
{
	if (!m_PipelineGroup) { return nullptr; }
	return m_PipelineGroup->get_context();
}

auto TestLib::Module::get_options() const noexcept -> const ModuleOptions&
{
	// TODO: return ステートメントをここに挿入します
	return m_Options;
}


auto TestLib::Module::get_opx7_module() const noexcept -> OptixModule {
	return m_Opx7Module;
}
