#include <RTLib/Ext/OPX7/OPX7Natives.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7Module.h>
#include <RTLib/Ext/OPX7/OPX7ProgramGroup.h>
#include <RTLib/Ext/OPX7/OPX7Pipeline.h>
#include <RTLib/Ext/OPX7/OPX7ShaderTable.h>

auto RTLib::Ext::OPX7::OPX7Natives::GetOptixDeviceContext(OPX7Context* context) -> OptixDeviceContext
{
    return context ? context->GetOptixDeviceContext() : nullptr;
}

auto RTLib::Ext::OPX7::OPX7Natives::GetOptixModule(OPX7Module* module) -> OptixModule
{
    return module ? module->GetOptixModule() : nullptr;
}

auto RTLib::Ext::OPX7::OPX7Natives::GetOptixProgramGroup(OPX7ProgramGroup* programGroup) -> OptixProgramGroup
{
    return programGroup ? programGroup->GetOptixProgramGroup() : nullptr;
}

auto RTLib::Ext::OPX7::OPX7Natives::GetOptixPipeline(OPX7Pipeline* pipeline) -> OptixPipeline
{
    return pipeline ? pipeline->GetOptixPipeline() : nullptr;
}
