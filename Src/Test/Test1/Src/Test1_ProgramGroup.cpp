#include <Test1_ProgramGroup.h>
#include <Test1_PipelineGroup.h>
#include <Test1_Module.h>
#include <iostream>
#include <optix_stubs.h>


Test1::ProgramGroup::ProgramGroup(std::string name, PipelineGroup* pipelineGroup, OptixProgramGroupKind kind)
	:m_Name{name},m_PipelineGroup{pipelineGroup},m_Kind{kind}
{
}

Test1::ProgramGroup::~ProgramGroup()
{
}


Test1::ProgramGroupRaygen::ProgramGroupRaygen(std::string name, PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName)
	:ProgramGroup(name, pipelineGroup, OPTIX_PROGRAM_GROUP_KIND_RAYGEN),m_EntryFunctionName{entryFunctionName},m_Module{module}
{
	auto context = pipelineGroup->get_context();
	auto opx7_context = context->get_opx7_device_context();

	OptixProgramGroupDesc desc = {};

	desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	desc.raygen.entryFunctionName = m_EntryFunctionName.c_str();
	desc.raygen.module = m_Module->get_opx7_module();

	OptixProgramGroupOptions options = {};

	char log[1024];
	size_t logSize = sizeof(log);
	OptixProgramGroup programGroups[2] = {};
	OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &desc, 1, &options, log, &logSize, &m_Opx7ProgramGroup));
	if (logSize != sizeof(log)) {
		std::cout << log << std::endl;
	}

}

Test1::ProgramGroupRaygen::~ProgramGroupRaygen() {
	OTK_ERROR_CHECK(optixProgramGroupDestroy(m_Opx7ProgramGroup));
	m_Opx7ProgramGroup = nullptr;
}

Test1::ProgramGroupMiss::ProgramGroupMiss(std::string name,PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName)
	:ProgramGroup(name,pipelineGroup, OPTIX_PROGRAM_GROUP_KIND_MISS), m_EntryFunctionName{ entryFunctionName }, m_Module{ module }
{
	auto context = pipelineGroup->get_context();
	auto opx7_context = context->get_opx7_device_context();

	OptixProgramGroupDesc desc = {};

	desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	desc.miss.entryFunctionName = m_EntryFunctionName.c_str();
	desc.miss.module = m_Module->get_opx7_module();

	OptixProgramGroupOptions options = {};

	char log[1024];
	size_t logSize = sizeof(log);
	OptixProgramGroup programGroups[2] = {};
	OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &desc, 1, &options, log, &logSize, &m_Opx7ProgramGroup));
	if (logSize != sizeof(log)) {
		std::cout << log << std::endl;
	}

}

Test1::ProgramGroupMiss::~ProgramGroupMiss() {
	OTK_ERROR_CHECK(optixProgramGroupDestroy(m_Opx7ProgramGroup));
	m_Opx7ProgramGroup = nullptr;
}

Test1::ProgramGroupHitgroup::ProgramGroupHitgroup(
	std::string name, PipelineGroup* pipelineGroup,
	Module* moduleCh, std::string entryFunctionNameCh,
	Module* moduleAh, std::string entryFunctionNameAh,
	Module* moduleIs, std::string entryFunctionNameIs
) :ProgramGroup(name,pipelineGroup, OPTIX_PROGRAM_GROUP_KIND_HITGROUP), 
	m_ModuleCH{ nullptr },
	m_ModuleAH{ nullptr },
	m_ModuleIS{ nullptr },
	m_EntryFunctionNameCH{ "" },
	m_EntryFunctionNameAH{ "" },
	m_EntryFunctionNameIS{ "" }
{
	auto context = pipelineGroup->get_context();
	auto opx7_context = context->get_opx7_device_context();

	OptixProgramGroupDesc desc = {};
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	if (!entryFunctionNameCh.empty() && moduleCh) {
		desc.hitgroup.entryFunctionNameCH = entryFunctionNameCh.c_str();
		desc.hitgroup.moduleCH = moduleCh->get_opx7_module();

		m_EntryFunctionNameCH = entryFunctionNameCh;
		m_ModuleCH = moduleCh;
	}
	if (!entryFunctionNameAh.empty() && moduleAh) {
		desc.hitgroup.entryFunctionNameAH = entryFunctionNameAh.c_str();
		desc.hitgroup.moduleAH = moduleAh->get_opx7_module();

		m_EntryFunctionNameAH = entryFunctionNameAh;
		m_ModuleAH = moduleAh;
	}
	if (!entryFunctionNameIs.empty() && moduleIs) {
		desc.hitgroup.entryFunctionNameIS = entryFunctionNameIs.c_str();
		desc.hitgroup.moduleIS = moduleIs->get_opx7_module();

		m_EntryFunctionNameIS = entryFunctionNameIs;
		m_ModuleIS = moduleIs;
	}


	OptixProgramGroupOptions options = {};

	char log[1024];
	size_t logSize = sizeof(log);
	OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &desc, 1, &options, log, &logSize, &m_Opx7ProgramGroup));
	if (logSize != sizeof(log)) {
		std::cout << log << std::endl;
	}

}

Test1::ProgramGroupHitgroup::~ProgramGroupHitgroup()
{
	OTK_ERROR_CHECK(optixProgramGroupDestroy(m_Opx7ProgramGroup));
	m_Opx7ProgramGroup = nullptr;
}

Test1::ProgramGroupCallable::ProgramGroupCallable(std::string name, PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName, bool supportTrace)
	:ProgramGroup(name, pipelineGroup, OPTIX_PROGRAM_GROUP_KIND_CALLABLES), m_EntryFunctionName{ entryFunctionName }, m_Module{ module },m_SupportTrace{supportTrace}
{
	auto context = pipelineGroup->get_context();
	auto opx7_context = context->get_opx7_device_context();

	OptixProgramGroupDesc desc = {};

	desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
	if (!supportTrace) {
		desc.callables.entryFunctionNameDC = m_EntryFunctionName.c_str();
		desc.callables.moduleDC = m_Module->get_opx7_module();
	}
	else {
		desc.callables.entryFunctionNameCC = m_EntryFunctionName.c_str();
		desc.callables.moduleCC = m_Module->get_opx7_module();
	}

	OptixProgramGroupOptions options = {};

	char log[1024];
	size_t logSize = sizeof(log);
	OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &desc, 1, &options, log, &logSize, &m_Opx7ProgramGroup));
	if (logSize != sizeof(log)) {
		std::cout << log << std::endl;
	}

}

Test1::ProgramGroupCallable::~ProgramGroupCallable() {
	OTK_ERROR_CHECK(optixProgramGroupDestroy(m_Opx7ProgramGroup));
	m_Opx7ProgramGroup = nullptr;
}
