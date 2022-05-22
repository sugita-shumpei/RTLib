#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7Module.h>
#include <RTLib/Ext/OPX7/OPX7ProgramGroup.h>
#include <RTLib/Ext/OPX7/OPX7Pipeline.h>
#include <RTLibExtOPX7TestConfig.h>
#include <memory>
#include <vector>
#include <fstream>
auto LoadBinary(const char* filename)->std::vector<char>
{
	auto binary = std::vector<char>();
	auto binaryFile = std::ifstream(filename, std::ios::binary);
	if (binaryFile.is_open()) {
		binaryFile.seekg(0, std::ios::end);
		auto size = static_cast<size_t>(binaryFile.tellg());
		binary.resize(size / sizeof(binary[0]));
		binaryFile.seekg(0, std::ios::beg);
		binaryFile.read((char*)binary.data(), size);
		binaryFile.close();
	}
	return binary;
}
int main() {
	RTLib::Ext::OPX7::OPX7ContextCreateDesc ctxDesc = {};
	ctxDesc.level          = RTLib::Ext::OPX7::OPX7ContextValidationLogLevel::ePrint;
	ctxDesc.pfnLogCallback = RTLib::Ext::OPX7::DefaultLogCallback;
	ctxDesc.pCallbackData  = nullptr;
	ctxDesc.validationMode = RTLib::Ext::OPX7::OPX7ContextValidationMode::eALL;
	auto context           = RTLib::Ext::OPX7::OPX7Context(ctxDesc);
	context.SetName("Opx7Context");
	context.Initialize();
	{
		RTLib::Ext::OPX7::OPX7PipelineCompileOptions pipeCompileOps = {};
		{

			pipeCompileOps.numPayloadValues          = 3;
			pipeCompileOps.numAttributeValues        = 3;
			pipeCompileOps.launchParamsVariableNames = "params";
			pipeCompileOps.usesPrimitiveTypeFlags    = 0;
			pipeCompileOps.traversableGraphFlags     = RTLib::Ext::OPX7::OPX7TraversableGraphFlagsAllowSingleGAS;
			pipeCompileOps.exceptionFlags            = 0;
			pipeCompileOps.usesMotionBlur            = false;
		}
		RTLib::Ext::OPX7::OPX7ModuleCreateDesc modDesc = {};
		{
			modDesc.moduleOptions.optLevel   = RTLib::Ext::OPX7::OPX7CompileOptimizationLevel::eDefault;
			modDesc.moduleOptions.debugLevel = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
			modDesc.moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
			modDesc.moduleOptions.boundValueEntries = {};
			modDesc.moduleOptions.payloadTypes = {};
			modDesc.pipelineOptions = pipeCompileOps;
			modDesc.ptxBinary = LoadBinary(RTLIB_EXT_OPX7_TEST_CUDA_PATH"/SimpleKernel.ptx");
		}
		auto opxModule = std::unique_ptr<RTLib::Ext::OPX7::OPX7Module>(context.CreateOPXModule(modDesc));
		auto pgDescs = std::vector<RTLib::Ext::OPX7::OPX7ProgramGroupCreateDesc>(3);
		{
			pgDescs[0].kind = RTLib::Ext::OPX7::OPX7ProgramGroupKind::eRayGen;
			pgDescs[0].raygen = {};
			pgDescs[0].raygen.entryFunctionName = "__raygen__simple_kernel";
			pgDescs[0].raygen.module = opxModule.get();

			pgDescs[1].kind = RTLib::Ext::OPX7::OPX7ProgramGroupKind::eMiss;
			pgDescs[1].miss = {};
			pgDescs[1].miss.entryFunctionName = "__miss__simple_kernel";
			pgDescs[1].miss.module = opxModule.get();

			pgDescs[2].kind = RTLib::Ext::OPX7::OPX7ProgramGroupKind::eMiss;
			pgDescs[2].hitgroup = {};
			pgDescs[2].hitgroup.closesthit.entryFunctionName = "__closesthit__simple_kernel";
			pgDescs[2].hitgroup.closesthit.module = opxModule.get();
		}
		auto opxProgramGroups  = context.CreateOPXProgramGroups(pgDescs);
		auto opxProgramGroupRG = std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>(opxProgramGroups[0]);
		auto opxProgramGroupMS = std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>(opxProgramGroups[1]);
		auto opxProgramGroupHG = std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>(opxProgramGroups[2]);

		auto pipeDesc = RTLib::Ext::OPX7::OPX7PipelineCreateDesc();
		{
			pipeDesc.compileOptions = pipeCompileOps;
			pipeDesc.linkOptions.debugLevel    = RTLib::Ext::OPX7::OPX7CompileDebugLevel::eDefault;
			pipeDesc.linkOptions.maxTraceDepth = 1;
			pipeDesc.programGroups             = opxProgramGroups;
		}
		auto opxPipeline = std::unique_ptr<RTLib::Ext::OPX7::OPX7Pipeline>(context.CreateOPXPipeline(pipeDesc));

		opxPipeline->Destroy();
		opxProgramGroupRG->Destroy();
		opxProgramGroupMS->Destroy();
		opxProgramGroupHG->Destroy();
		opxModule->Destroy();
	}
	context.Terminate();
	return 0;
}