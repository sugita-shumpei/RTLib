#include <RTLib/Ext/OPX7/OPX7ProgramGroup.h>
#include <RTLib/Ext/OPX7/OPX7Module.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7Exceptions.h>
#include <optix_stubs.h>
#include <cassert>
struct RTLib::Ext::OPX7::OPX7ProgramGroup::Impl
{
    OPX7Context*      context = nullptr;
    OptixProgramGroup programGroup   = nullptr;
    OPX7ProgramGroupKind kind = OPX7ProgramGroupKind::eRayGen;
};
auto RTLib::Ext::OPX7::OPX7ProgramGroup::Enumerate(OPX7Context *context, const std::vector<OPX7ProgramGroupCreateDesc> &descs, const OPX7ProgramGroupOptions& options) -> std::vector<OPX7ProgramGroup *>
{
    if (!context) { return std::vector<OPX7ProgramGroup*>(); }
    std::vector<OPX7ProgramGroup*>     programGroups(descs.size(), nullptr);
    std::vector<OptixProgramGroupDesc> opxProgramGroupDescs(descs.size());
    std::vector<OptixProgramGroup>     opxProgramGroups(descs.size(), nullptr);
    OptixProgramGroupOptions           opxProgramGroupOptions;
    auto deviceContext = context->GetOptixDeviceContext();
    auto payloadType = options.payloadType;
    opxProgramGroupOptions.payloadType = payloadType ? &payloadType.value() : nullptr;
    for (size_t i = 0; i < descs.size(); ++i)
    {
        auto kind = static_cast<OptixProgramGroupKind>(descs[i].kind);
        opxProgramGroupDescs[i].kind = kind;
        switch (kind)
        {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            opxProgramGroupDescs[i].raygen = {};
            opxProgramGroupDescs[i].raygen.entryFunctionName = descs[i].raygen.entryFunctionName;
            opxProgramGroupDescs[i].raygen.module = descs[i].raygen.module ? descs[i].raygen.module->GetOptixModule(): nullptr;
            break;
        case OPTIX_PROGRAM_GROUP_KIND_MISS:
            opxProgramGroupDescs[i].miss = {};
            opxProgramGroupDescs[i].miss.entryFunctionName = descs[i].miss.entryFunctionName;
            opxProgramGroupDescs[i].miss.module = descs[i].miss.module ? descs[i].miss.module->GetOptixModule() : nullptr;
            break;
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            opxProgramGroupDescs[i].exception = {};
            opxProgramGroupDescs[i].exception.entryFunctionName = descs[i].exception.entryFunctionName;
            opxProgramGroupDescs[i].exception.module = descs[i].exception.module ? descs[i].exception.module->GetOptixModule() : nullptr;
            break;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            opxProgramGroupDescs[i].hitgroup = {};
            if (descs[i].hitgroup.intersect.module) {
                opxProgramGroupDescs[i].hitgroup.entryFunctionNameIS = descs[i].hitgroup.intersect.entryFunctionName;
                opxProgramGroupDescs[i].hitgroup.moduleIS = descs[i].hitgroup.intersect.module->GetOptixModule();
            }
            if (descs[i].hitgroup.closesthit.module) {
                opxProgramGroupDescs[i].hitgroup.entryFunctionNameCH = descs[i].hitgroup.closesthit.entryFunctionName;
                opxProgramGroupDescs[i].hitgroup.moduleCH = descs[i].hitgroup.closesthit.module->GetOptixModule();
            }
            if (descs[i].hitgroup.anyhit.module) {
                opxProgramGroupDescs[i].hitgroup.entryFunctionNameAH = descs[i].hitgroup.anyhit.entryFunctionName;
                opxProgramGroupDescs[i].hitgroup.moduleAH = descs[i].hitgroup.anyhit.module->GetOptixModule();
            }
            break;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            opxProgramGroupDescs[i].callables = {};
            if (descs[i].callables.direct.module) {
                opxProgramGroupDescs[i].callables.entryFunctionNameDC = descs[i].callables.direct.entryFunctionName;
                opxProgramGroupDescs[i].callables.moduleDC            = descs[i].callables.direct.module->GetOptixModule();
            }
            if (descs[i].callables.continuation.module) {
                opxProgramGroupDescs[i].callables.entryFunctionNameCC = descs[i].callables.continuation.entryFunctionName;
                opxProgramGroupDescs[i].callables.moduleCC            = descs[i].callables.continuation.module->GetOptixModule();
            }
            break;
        default:
            break;
        }
    }
    char logString[1024];
    size_t length = sizeof(logString);
    RTLIB_EXT_OPX7_THROW_IF_FAILED(optixProgramGroupCreate(
        deviceContext,
        opxProgramGroupDescs.data(),
        static_cast<unsigned int>(opxProgramGroupDescs.size()),
        &opxProgramGroupOptions,
        logString,&length, opxProgramGroups.data()
    ));
    for (size_t i = 0; i < descs.size(); ++i) {
        programGroups[i] = new OPX7ProgramGroup();
        programGroups[i]->m_Impl->context = context;
        programGroups[i]->m_Impl->programGroup  = opxProgramGroups[i];
        programGroups[i]->m_Impl->kind          = descs[i].kind;
    }
    return programGroups;
}

RTLib::Ext::OPX7::OPX7ProgramGroup::~OPX7ProgramGroup() noexcept
{
    m_Impl.reset();
}

void RTLib::Ext::OPX7::OPX7ProgramGroup::Destroy()
{
    if (!m_Impl) { return; }
    if (!m_Impl->programGroup) { return; }
    try {
        RTLIB_EXT_OPX7_THROW_IF_FAILED(optixProgramGroupDestroy(m_Impl->programGroup));
    }
    catch (OPX7Exception& err) {
        std::cerr << err.what() << std::endl;
    }
    m_Impl->programGroup = nullptr;
}

auto RTLib::Ext::OPX7::OPX7ProgramGroup::GetKind() const noexcept -> OPX7ProgramGroupKind
{
    assert(m_Impl != nullptr);
    return m_Impl->kind;
}

RTLib::Ext::OPX7::OPX7ProgramGroup::OPX7ProgramGroup() noexcept : m_Impl{new Impl()}
{
}

auto RTLib::Ext::OPX7::OPX7ProgramGroup::GetOptixProgramGroup() const noexcept -> OptixProgramGroup
{
    assert(m_Impl != nullptr);
    return m_Impl->programGroup;
}
