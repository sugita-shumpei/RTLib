#include <Test1_Module.h>
#include <optix_stubs.h>

Test1::Module::Module(std::string name,Test1::PipelineGroup* pipeline_group, const ModuleOptions& options, const char* ptx, size_t ptxSize)
	:m_PipelineGroup{pipeline_group},m_Options{options}, m_Module{nullptr},m_Name{name}
{
	auto moduleCompileOptions = m_Options.get_opx7_module_compile_options();
	auto context = m_PipelineGroup->get_context();
	auto pipelineCompileOptions = m_PipelineGroup->get_options();
	auto opx7DeviceContext = context->get_opx7_device_context();

	char log[1024];
	size_t logSize = sizeof(log);

	OTK_ERROR_CHECK(optixModuleCreateFromPTX(opx7DeviceContext, &moduleCompileOptions, &pipelineCompileOptions, ptx, ptxSize, log, &logSize, &m_Module));
	if (logSize != sizeof(log)) {
		std::cerr << log << std::endl;
	}
}

Test1::Module::~Module()
{
	OTK_ERROR_CHECK(optixModuleDestroy(m_Module));
	m_Module = nullptr;
}

auto Test1::Module::get_name() const noexcept -> const char*
{
	return m_Name.c_str();
}

auto Test1::Module::get_pipeline_group() const noexcept -> const Test1::PipelineGroup*
{
	return m_PipelineGroup;
}

auto Test1::Module::get_pipeline_group() noexcept -> Test1::PipelineGroup*
{
	return m_PipelineGroup;
}

auto Test1::Module::get_context() const noexcept ->const Test1::Context*
{
	if (!m_PipelineGroup) { return nullptr; }
	return m_PipelineGroup->get_context();
}

auto Test1::Module::get_options() const noexcept -> const ModuleOptions&
{
	// TODO: return ステートメントをここに挿入します
	return m_Options;
}


auto Test1::Module::get_opx7_module() const noexcept -> OptixModule {
	return m_Module;
}