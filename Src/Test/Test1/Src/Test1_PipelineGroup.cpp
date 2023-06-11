#include <Test1_PipelineGroup.h>
#include <Test1_Module.h>
#include <Test1_ProgramGroup.h>
#include <Test1_Pipeline.h>

Test1::PipelineGroup::PipelineGroup(const Test1::Context* context, const OptixPipelineCompileOptions& options)
	:m_Context{context},m_Options{ options }
{
}

Test1::PipelineGroup::~PipelineGroup()
{
	m_Modules.clear();
}

auto Test1::PipelineGroup::get_context() const noexcept -> const Test1::Context*
{
	// TODO: return ステートメントをここに挿入します
	return m_Context;
}

auto Test1::PipelineGroup::get_options() const noexcept -> const OptixPipelineCompileOptions&
{
	// TODO: return ステートメントをここに挿入します
	return m_Options;
}

bool Test1::PipelineGroup::load_module(std::string name, const ModuleOptions& options, const char* ptx, size_t ptxSize)
{
	try {
		m_Modules[name] = std::unique_ptr<Module>(new Module(name,this, options, ptx, ptxSize));
		return true;
	}
	catch (std::runtime_error& err)
	{
		std::cerr << "Failed To Load Module: " << name << std::endl;
		return false;
	}
}

auto Test1::PipelineGroup::get_module(std::string name) const noexcept -> Module*
{
	if (m_Modules.count(name) == 0) {
		return nullptr;
	}
	return m_Modules.at(name).get();
}

bool Test1::PipelineGroup::has_module(std::string name) const noexcept
{
	return m_Modules.count(name) != 0;
}

bool Test1::PipelineGroup::load_program_group_rg(std::string pgName, std::string moduleName, std::string entryName)
{
	auto module = get_module(moduleName);
	if (!module) { return false; }
	if (entryName.empty()) { return false; }

	OptixProgramGroupDesc desc = {};
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	entryName = "__raygen__" + entryName;
	desc.raygen.entryFunctionName = entryName.c_str();
	desc.raygen.module = module->get_opx7_module();

	m_ProgramGroupsRG[pgName].reset(new ProgramGroupRaygen(pgName,this, module, entryName));

	return true;
}

bool Test1::PipelineGroup::load_program_group_ms(std::string pgName, std::string moduleName, std::string entryName)
{
	auto module = get_module(moduleName);
	if (!module) { return false; }
	if (entryName.empty()) { return false; }

	OptixProgramGroupDesc desc = {};
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	entryName = "__miss__" + entryName;
	desc.miss.entryFunctionName = entryName.c_str();
	desc.miss.module = module->get_opx7_module();

	m_ProgramGroupsMS[pgName].reset(new ProgramGroupMiss(pgName,this, module, entryName));

	return true;
}

bool Test1::PipelineGroup::load_program_group_hg(std::string pgName, std::string moduleNameCh, std::string entryNameCh, std::string moduleNameAh, std::string entryNameAh, std::string moduleNameIs, std::string entryNameIs)
{

	OptixProgramGroupDesc desc = {};
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	auto moduleCh = get_module(moduleNameCh);
	if (!moduleCh) {
		entryNameCh = "";
	}
	else
	{
		entryNameCh = "__closesthit__" + entryNameCh;
	}

	auto moduleAh = get_module(moduleNameAh);
	if (!moduleAh) { 
		entryNameAh = "";
	}
	else
	{
		entryNameAh = "__anyhit__" + entryNameAh;
	}

	auto moduleIs = get_module(moduleNameIs);
	auto opx7ModuleIs = OptixModule(nullptr);
	if (!moduleIs) {
		entryNameIs = "";
	}
	else
	{
		entryNameIs = "__intersection__" + entryNameIs;
	}

	m_ProgramGroupsHG[pgName].reset(new ProgramGroupHitgroup(pgName,this, moduleCh, entryNameCh,moduleAh,entryNameAh,moduleIs,entryNameIs));

	return true;
}

bool Test1::PipelineGroup::load_program_group_dc(std::string pgName, std::string moduleName, std::string entryName)
{
	auto module = get_module(moduleName);
	if (!module) { return false; }
	if (entryName.empty()) { return false; }

	entryName = "__direct_callable__ " + entryName;

	m_ProgramGroupsDC[pgName].reset(new ProgramGroupCallable(pgName,this, module, entryName,false));
	return true;
}

bool Test1::PipelineGroup::load_program_group_cc(std::string pgName, std::string moduleName, std::string entryName)
{
	auto module = get_module(moduleName);
	if (!module) { return false; }
	if (entryName.empty()) { return false; }

	OptixProgramGroupDesc desc = {};
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
	entryName = "__continuation_callable__ " + entryName;

	m_ProgramGroupsCC[pgName].reset(new ProgramGroupCallable(pgName,this, module, entryName, false));
	return true;
}

auto Test1::PipelineGroup::get_program_group_rg(std::string pgName) const noexcept -> ProgramGroup*
{
	if (m_ProgramGroupsRG.count(pgName) == 0) { return nullptr; }
	return m_ProgramGroupsRG.at(pgName).get();
}

auto Test1::PipelineGroup::get_program_group_ms(std::string pgName) const noexcept -> ProgramGroup*
{

	if (m_ProgramGroupsMS.count(pgName) == 0) { return nullptr; }
	return m_ProgramGroupsMS.at(pgName).get();
}

auto Test1::PipelineGroup::get_program_group_hg(std::string pgName) const noexcept -> ProgramGroup*
{
	if (m_ProgramGroupsHG.count(pgName) == 0) { return nullptr; }
	return m_ProgramGroupsHG.at(pgName).get();
}

auto Test1::PipelineGroup::get_program_group_dc(std::string pgName) const noexcept -> ProgramGroup*
{
	if (m_ProgramGroupsDC.count(pgName) == 0) { return nullptr; }
	return m_ProgramGroupsDC.at(pgName).get();
}

auto Test1::PipelineGroup::get_program_group_cc(std::string pgName) const noexcept -> ProgramGroup*
{
	if (m_ProgramGroupsCC.count(pgName) == 0) { return nullptr; }
	return m_ProgramGroupsCC.at(pgName).get();
}

bool Test1::PipelineGroup::has_program_group_rg(std::string pgName) const noexcept
{
	return m_ProgramGroupsRG.count(pgName)!=0;
}

bool Test1::PipelineGroup::has_program_group_ms(std::string pgName) const noexcept
{
	return m_ProgramGroupsMS.count(pgName) != 0;
}

bool Test1::PipelineGroup::has_program_group_hg(std::string pgName) const noexcept
{
	return m_ProgramGroupsHG.count(pgName) != 0;
}

bool Test1::PipelineGroup::has_program_group_dc(std::string pgName) const noexcept
{
	return m_ProgramGroupsDC.count(pgName) != 0;
}

bool Test1::PipelineGroup::has_program_group_cc(std::string pgName) const noexcept
{
	return m_ProgramGroupsCC.count(pgName) != 0;
}

bool Test1::PipelineGroup::load_pipeline(std::string pipelineName, std::string pgNameRg, const std::vector<std::string>& pgNamesMs, const std::vector<std::string>& pgNamesHg, const std::vector<std::string>& pgNamesDc, const std::vector<std::string>& pgNamesCc, OptixCompileDebugLevel debugLevel, unsigned int maxTraceDepth)
{
	
	ProgramGroup* programGroupRg = nullptr;
	if (!pgNameRg.empty())
	{
		programGroupRg = m_ProgramGroupsRG.at(pgNameRg).get();
	}
	std::vector<ProgramGroup*> programGroupsMs = {};
	for (auto& pgNameMs : pgNamesMs)
	{
		if (pgNameMs.empty()) { continue; }
		auto& programGroupMs = m_ProgramGroupsMS.at(pgNameMs);
		programGroupsMs.push_back(programGroupMs.get());
	}
	std::vector<ProgramGroup*> programGroupsHg = {};
	for (auto& pgNameHg : pgNamesHg)
	{
		if (pgNameHg.empty()) { continue; }
		auto& programGroupHg = m_ProgramGroupsHG.at(pgNameHg);
		programGroupsHg.push_back(programGroupHg.get());
	}
	std::vector<ProgramGroup*> programGroupsDc = {};
	for (auto& pgNameDc : pgNamesDc)
	{
		if (pgNameDc.empty()) { continue; }
		auto& programGroupDC = m_ProgramGroupsDC.at(pgNameDc);
		programGroupsDc.push_back(programGroupDC.get());
	}
	std::vector<ProgramGroup*> programGroupsCc = {};
	for (auto& pgNameCc : pgNamesCc)
	{
		if (pgNameCc.empty()) { continue; }
		auto& programGroupCC = m_ProgramGroupsCC.at(pgNameCc);
		programGroupsCc.push_back(programGroupCC.get());
	}
	OptixPipelineLinkOptions linkOptions = {};
	linkOptions.debugLevel = debugLevel;
	linkOptions.maxTraceDepth = maxTraceDepth;
	m_Pipelines[pipelineName].reset(new Pipeline(pipelineName, this,
		programGroupRg, programGroupsMs, programGroupsHg, programGroupsDc, programGroupsCc,
		linkOptions
	));
	return true;

}

auto Test1::PipelineGroup::get_pipeline(std::string pipelineName) const noexcept -> Pipeline*
{
	if (m_Pipelines.count(pipelineName) == 0) { return nullptr; }
	return m_Pipelines.at(pipelineName).get();
}

bool Test1::PipelineGroup::has_pipeline(std::string pipelineName) const noexcept
{
	return m_Pipelines.count(pipelineName) != 0;
}
