#include <TestLib/ProgramGroup.h>
#include <TestLib/PipelineGroup.h>
#include <TestLib/Module.h>
#include <iostream>
#include <optix_stubs.h>


TestLib::ProgramGroup::ProgramGroup(std::string name, PipelineGroup* pipelineGroup, OptixProgramGroupKind kind)
	:m_Name{name},
	m_PipelineGroup{pipelineGroup},
	m_Kind{kind}
{
}

TestLib::ProgramGroup::~ProgramGroup()
{
}


TestLib::ProgramGroupRaygen::ProgramGroupRaygen(std::string name, PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName)
	:ProgramGroup(name, pipelineGroup, OPTIX_PROGRAM_GROUP_KIND_RAYGEN), 
	m_EntryFunctionName{ entryFunctionName }, 
	m_Opx7Module{ module }, 
	m_StackSizes{}
{
	auto context = pipelineGroup->get_context();
	auto opx7_context = context->get_opx7_device_context();

	OptixProgramGroupDesc desc = {};

	desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	desc.raygen.entryFunctionName = m_EntryFunctionName.c_str();
	desc.raygen.module = m_Opx7Module->get_opx7_module();

	OptixProgramGroupOptions options = {};

	char log[1024];
	size_t logSize = sizeof(log);
	OptixProgramGroup programGroups[2] = {};
	OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &desc, 1, &options, log, &logSize, &m_Opx7ProgramGroup));
	if (logSize != sizeof(log)) {
		std::cout << log << std::endl;
	}
	OTK_ERROR_CHECK(optixProgramGroupGetStackSize(m_Opx7ProgramGroup, &m_StackSizes));
}

TestLib::ProgramGroupRaygen::~ProgramGroupRaygen() {
	OTK_ERROR_CHECK(optixProgramGroupDestroy(m_Opx7ProgramGroup));
	m_Opx7ProgramGroup = nullptr;
}

TestLib::ProgramGroupMiss::ProgramGroupMiss(std::string name,PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName)
	:ProgramGroup(name,pipelineGroup, OPTIX_PROGRAM_GROUP_KIND_MISS), 
	m_EntryFunctionName{ entryFunctionName }, 
	m_Opx7Module{ module }, 
	m_StackSizes{}
{
	auto context = pipelineGroup->get_context();
	auto opx7_context = context->get_opx7_device_context();

	OptixProgramGroupDesc desc = {};

	desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	desc.miss.entryFunctionName = m_EntryFunctionName.c_str();
	desc.miss.module = m_Opx7Module->get_opx7_module();

	OptixProgramGroupOptions options = {};

	char log[1024];
	size_t logSize = sizeof(log);
	OptixProgramGroup programGroups[2] = {};
	OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &desc, 1, &options, log, &logSize, &m_Opx7ProgramGroup));
	if (logSize != sizeof(log)) {
		std::cout << log << std::endl;
	}
	OTK_ERROR_CHECK(optixProgramGroupGetStackSize(m_Opx7ProgramGroup, &m_StackSizes));

}

TestLib::ProgramGroupMiss::~ProgramGroupMiss() {
	OTK_ERROR_CHECK(optixProgramGroupDestroy(m_Opx7ProgramGroup));
	m_Opx7ProgramGroup = nullptr;
}

TestLib::ProgramGroupHitgroup::ProgramGroupHitgroup(
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
	m_EntryFunctionNameIS{ "" }, 
	m_StackSizes{}
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
	if (moduleIs) {
		if (entryFunctionNameIs != "") {
			desc.hitgroup.entryFunctionNameIS = entryFunctionNameIs.c_str();
		}
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
	OTK_ERROR_CHECK(optixProgramGroupGetStackSize(m_Opx7ProgramGroup, &m_StackSizes));

}

TestLib::ProgramGroupHitgroup::~ProgramGroupHitgroup()
{
	OTK_ERROR_CHECK(optixProgramGroupDestroy(m_Opx7ProgramGroup));
	m_Opx7ProgramGroup = nullptr;
}

TestLib::ProgramGroupCallable::ProgramGroupCallable(std::string name, PipelineGroup* pipelineGroup, Module* module, std::string entryFunctionName, bool supportTrace)
	:ProgramGroup(name, pipelineGroup, OPTIX_PROGRAM_GROUP_KIND_CALLABLES), 
	m_EntryFunctionName{ entryFunctionName }, 
	m_Opx7Module{ module },
	m_SupportTrace{supportTrace}, 
	m_StackSizes{}
{
	auto context = pipelineGroup->get_context();
	auto opx7_context = context->get_opx7_device_context();

	OptixProgramGroupDesc desc = {};

	desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
	if (!supportTrace) {
		desc.callables.entryFunctionNameDC = m_EntryFunctionName.c_str();
		desc.callables.moduleDC = m_Opx7Module->get_opx7_module();
	}
	else {
		desc.callables.entryFunctionNameCC = m_EntryFunctionName.c_str();
		desc.callables.moduleCC = m_Opx7Module->get_opx7_module();
	}

	OptixProgramGroupOptions options = {};

	char log[1024];
	size_t logSize = sizeof(log);
	OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &desc, 1, &options, log, &logSize, &m_Opx7ProgramGroup));
	if (logSize != sizeof(log)) {
		std::cout << log << std::endl;
	}
	OTK_ERROR_CHECK(optixProgramGroupGetStackSize(m_Opx7ProgramGroup, &m_StackSizes));

}

TestLib::ProgramGroupCallable::~ProgramGroupCallable() {
	OTK_ERROR_CHECK(optixProgramGroupDestroy(m_Opx7ProgramGroup));
	m_Opx7ProgramGroup = nullptr;
}
